import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import lorentz as L


def soft_clip(x, r=8.0):
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return x * (r * torch.tanh(n / r)) / n


def ext_angle(x, y, curv):
    return torch.nan_to_num(L.oxy_angle(x, y, curv, eps=1e-4), nan=0.0, posinf=math.pi, neginf=0.0)


def aperture(x, curv):
    return torch.nan_to_num(L.half_aperture(x, curv, eps=1e-4), nan=0.0, posinf=math.pi / 2, neginf=0.0)


class Modality(nn.Module):
    def __init__(self, encoder, feat_dim, num_class, embed_dim, rel_dim, reliability):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_class))
        self.proj = nn.Identity() if feat_dim == embed_dim else nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.GELU())
        self.log_alpha = nn.Parameter(torch.tensor(-2.0))     
        self.reliability = reliability
        if reliability:
            self.rel = nn.Linear(feat_dim, rel_dim)
            self.rel_log_beta = nn.Parameter(torch.tensor(0.0))


class HyRAF(nn.Module):
    def __init__(self, modalities, num_class, align, distill=(), task='multiclass', embed_dim=512, rel_dim=8,
                 clip_r=8.0, queue=1024, queue_warmup=1200, kd_T=2.0, geom_thr=3.0, ent_delta=0.3, sev_margin=2.0,
                 modality_dropout=0.15, ema_momentum=0.99,
                 w_align=0.5, w_branch=0.2, w_proto=0.3, w_kd=1.0, w_rel=0.3):
        super().__init__()
        self.names = list(modalities)
        self.mods = nn.ModuleDict({n: Modality(e, d, num_class, embed_dim, rel_dim, r) for n, (e, d, r) in modalities.items()})
        self.num_class, self.task, self.align, self.distill = num_class, task, list(align), list(distill)
        self.clip_r, self.kd_T, self.geom_thr, self.ent_delta, self.sev_margin = clip_r, kd_T, geom_thr, ent_delta, sev_margin
        self.modality_dropout, self.ema_momentum, self.queue_warmup = modality_dropout, ema_momentum, queue_warmup
        self.w = dict(align=w_align, branch=w_branch, proto=w_proto, kd=w_kd, rel=w_rel)
        M = len(self.names)
        # alignment hyperboloid
        self.log_curv = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        # reliability hyperboloid
        self.rel_log_curv = nn.Parameter(torch.tensor(-2.0))
        # GeoWeight: signed per-modality coefficients on the standardised radius and angle
        self.gw_radius = nn.Parameter(torch.zeros(M))
        self.gw_angle = nn.Parameter(torch.zeros(M))
        # prototype branch + confidence gate
        self.proto = nn.Parameter(torch.randn(num_class, embed_dim) * 0.01)
        self.proto_log_w = nn.Parameter(torch.tensor(math.log(0.5)))
        self.fuse_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(0.2))
        self.conf_cos = nn.Parameter(torch.tensor(2.0)); self.conf_ang = nn.Parameter(torch.tensor(1.0)); self.conf_bias = nn.Parameter(torch.tensor(0.0))
        # running clean statistics (log-radius and anchor angle in the reliability space) and clean class anchors
        self.register_buffer('stat_init', torch.zeros(M, dtype=torch.bool))
        self.register_buffer('logr_mu', torch.zeros(M)); self.register_buffer('logr_var', torch.ones(M))
        self.register_buffer('ang_mu', torch.zeros(M)); self.register_buffer('ang_var', torch.ones(M))
        self.register_buffer('anchor', torch.zeros(M, num_class, rel_dim)); self.register_buffer('anchor_init', torch.zeros(M, num_class, dtype=torch.bool))
        # memory queues for the alignment (one per pair)
        self.queue = queue
        for k in range(len(self.align)):
            self.register_buffer(f'q{k}_a', torch.zeros(queue, embed_dim)); self.register_buffer(f'q{k}_b', torch.zeros(queue, embed_dim))
        self.register_buffer('q_ptr', torch.zeros(len(self.align), dtype=torch.long)); self.register_buffer('q_fill', torch.zeros(len(self.align), dtype=torch.long))
        self.register_buffer('steps', torch.zeros((), dtype=torch.long))

    #  geometry
    def curv(self):
        self.log_curv.data.clamp_(math.log(0.01), 4.0); return self.log_curv.exp()

    def rel_curv(self):
        self.rel_log_curv.data.clamp_(-2.0, 4.0); return self.rel_log_curv.exp()

    def embed(self, n, h):
        m = self.mods[n]; m.log_alpha.data.clamp_(max=0.0)
        return L.exp_map0(soft_clip(m.proj(h).float() * m.log_alpha.exp(), self.clip_r), self.curv())

    def rel_embed(self, n, h):
        m = self.mods[n]
        return L.exp_map0(soft_clip(m.rel(h.float()) * m.rel_log_beta.exp(), self.clip_r), self.rel_curv())

    def _mean_anchor(self, i):
        ok = self.anchor_init[i]
        return self.anchor[i][ok].mean(0) if bool(ok.any()) else None

    def _signals(self, i, u):
        lr = torch.log(u.norm(dim=-1).clamp(min=1e-4))
        r = (lr - self.logr_mu[i]) / self.logr_var[i].sqrt()
        a0 = self._mean_anchor(i)
        if a0 is None: return r, torch.zeros_like(r), lr, torch.zeros_like(r)
        ang = ext_angle(u, a0.expand_as(u), self.rel_curv())
        return r, (ang - self.ang_mu[i]) / self.ang_var[i].sqrt(), lr, ang

    @torch.no_grad()
    def _update_stats(self, i, lr, ang, u, y, clean):
        if not bool(clean.any()): return
        m = self.ema_momentum
        for buf_mu, buf_var, v in ((self.logr_mu, self.logr_var, lr[clean]), (self.ang_mu, self.ang_var, ang[clean])):
            mu, var = v.mean(), v.var(unbiased=False) + 1e-6
            if not bool(self.stat_init[i]): buf_mu[i], buf_var[i] = mu, var
            else: buf_mu[i] = m * buf_mu[i] + (1 - m) * mu; buf_var[i] = m * buf_var[i] + (1 - m) * var
        self.stat_init[i] = True
        if self.task == 'multiclass':
            for c in range(self.num_class):
                k = clean & (y == c)
                if bool(k.any()):
                    v = u[k].mean(0)
                    self.anchor[i, c] = v if not bool(self.anchor_init[i, c]) else m * self.anchor[i, c] + (1 - m) * v
                    self.anchor_init[i, c] = True
        else:   # multilabel: a single clean anchor stored in slot 0
            v = u[clean].mean(0)
            self.anchor[i, 0] = v if not bool(self.anchor_init[i, 0]) else m * self.anchor[i, 0] + (1 - m) * v
            self.anchor_init[i, 0] = True

    def _encode_clean(self, n, xc):
        enc = self.mods[n].encoder
        if xc.shape[0] > 1 or not self.training: return enc(xc)
        bns = [m for m in enc.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for m in bns: m.eval()
        try: return enc(xc)
        finally:
            for m in bns: m.train()

    # losses
    def _cls_loss(self, z, y, weight=None):
        if self.task == 'multiclass': return F.cross_entropy(z, y, weight=weight)
        return F.binary_cross_entropy_with_logits(z, y.float(), pos_weight=weight)

    def _kd(self, zs, zt):
        T = self.kd_T; zt = zt.detach()
        if self.task == 'multiclass':
            return F.kl_div(F.log_softmax(zs / T, -1), F.softmax(zt / T, -1), reduction='batchmean') * T * T
        return F.binary_cross_entropy_with_logits(zs / T, torch.sigmoid(zt / T)) * T * T

    def _align_loss(self, k, ea, eb, ok):
        n = int(ok.sum())
        if n < 2: return ea.new_zeros(())
        c = self.curv(); self.log_scale.data.clamp_(max=math.log(100.0)); s = self.log_scale.exp()
        a, b = ea[ok], eb[ok]
        K = int(self.q_fill[k]) if int(self.steps) >= self.queue_warmup else 0
        qa, qb = getattr(self, f'q{k}_a')[:K], getattr(self, f'q{k}_b')[:K]
        la = -s * L.pairwise_dist(a, torch.cat([b, qb], 0), c)
        lb = -s * L.pairwise_dist(b, torch.cat([a, qa], 0), c)
        t = torch.arange(n, device=a.device)
        loss = 0.5 * (F.cross_entropy(la, t) + F.cross_entropy(lb, t))
        if K > 0: loss = loss * math.log(n) / math.log(n + K)          
        if self.queue == 0: return loss                                 
        with torch.no_grad():
            Q = self.queue; p = int(self.q_ptr[k]); idx = (torch.arange(n, device=a.device) + p) % Q
            getattr(self, f'q{k}_a')[idx] = a.detach(); getattr(self, f'q{k}_b')[idx] = b.detach()
            self.q_ptr[k] = (p + n) % Q; self.q_fill[k] = min(Q, int(self.q_fill[k]) + n)
        return loss

    def _rel_loss(self, u_deg, u_clean, sev):
        c = self.rel_curv()
        cone = F.relu(ext_angle(u_deg, u_clean, c) - aperture(u_deg, c) - self.ent_delta).mean()
        d = torch.log(u_clean.norm(dim=-1).clamp(min=1e-4)) - torch.log(u_deg.norm(dim=-1).clamp(min=1e-4))
        return cone + (d - self.sev_margin * sev).abs().mean()

    # forward
    def forward(self, x, avail=None, y=None, clean=None, severity=None, class_weight=None):
        dev = next(iter(x.values())).device; B = next(iter(x.values())).shape[0]
        names = self.names
        av = torch.stack([(avail[n] if avail is not None and n in avail else torch.ones(B, dtype=torch.bool, device=dev)).bool() for n in names], 1)
        h = {n: self.mods[n].encoder(x[n]) for n in names}
        z = {n: self.mods[n].head(h[n]) for n in names}
        e = {n: self.embed(n, h[n]) for n in names}
        u = {n: self.rel_embed(n, h[n]) for n in names if self.mods[n].reliability}
        deg = {n: (severity[n] >= 0) if severity is not None and n in severity else torch.zeros(B, dtype=torch.bool, device=dev) for n in names}

        # GeoWeight (detached geometry signals, learnable signed coefficients)
        score = torch.zeros(B, len(names), device=dev); sig = {}
        with torch.no_grad():
            for i, n in enumerate(names):
                if n in u: sig[n] = self._signals(i, u[n].detach())
        for i, n in enumerate(names):
            if n in sig: score[:, i] = self.gw_radius[i] * sig[n][0].clamp(-6, 6) + self.gw_angle[i] * sig[n][1].clamp(-6, 6)
        active = av.clone()
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(B, len(names), device=dev) < self.modality_dropout
            keep = active & ~drop
            none_left = keep.sum(1) == 0
            keep[none_left] = active[none_left]
            active = keep
        wts = torch.softmax(score.masked_fill(~active, -1e9), dim=1)
        z_fused = sum(wts[:, i:i + 1] * z[n] for i, n in enumerate(names))

        # prototype branch on the GeoWeight-fused embedding, scaled by a cross-modal confidence gate
        c = self.curv()
        t = self.fuse_head(sum(wts[:, i:i + 1] * self.mods[n].proj(h[n]) for i, n in enumerate(names)))
        f = L.exp_map0(soft_clip(t.float(), self.clip_r), c)
        P = L.exp_map0(soft_clip(self.proto, self.clip_r), c)
        fr, pr = f.unsqueeze(1).expand(B, self.num_class, -1), P.unsqueeze(0).expand(B, self.num_class, -1)
        viol = ext_angle(pr.reshape(-1, P.shape[1]), fr.reshape(-1, P.shape[1]), c) - aperture(pr.reshape(-1, P.shape[1]), c)
        z_proto = -viol.view(B, self.num_class) * F.softplus(self.proto_log_w)
        with torch.no_grad():   # cross-modal agreement, averaged over the aligned pairs present in each row
            cos_m = torch.zeros(B, device=dev); ang_m = torch.zeros(B, device=dev); cnt = torch.zeros(B, device=dev)
            for a_, b_ in self.align:
                ia, ib = names.index(a_), names.index(b_); both = (active[:, ia] & active[:, ib]).float()
                cos_m += both * F.cosine_similarity(self.mods[a_].proj(h[a_]), self.mods[b_].proj(h[b_]), dim=-1).float()
                ang_m += both * ext_angle(e[a_], e[b_], c).clamp(0, math.pi)
                cnt += both
            cos_m = cos_m / cnt.clamp(min=1); ang_m = ang_m / cnt.clamp(min=1)
        conf = torch.sigmoid(self.conf_cos.clamp(min=0.1) * cos_m - self.conf_ang.clamp(min=0.1) * ang_m + self.conf_bias)
        gate = conf.clamp(0.25, 1.25).unsqueeze(-1)
        logits = z_fused + gate * z_proto

        out = {'weights': wts.detach(), 'branch_logits': {n: z[n].detach() for n in names}}
        if y is None:
            out['reliability'] = {n: (-sig[n][0]).detach() for n in sig}     
            return logits, out

        # training losses
        loss_cls = self._cls_loss(logits, y, class_weight)
        loss_branch = sum(self._cls_loss(z[n][av[:, i]], y[av[:, i]]) if bool(av[:, i].any()) else logits.new_zeros(())
                          for i, n in enumerate(names)) / len(names)
        loss_proto = self._cls_loss(z_proto, y)
        loss_align = logits.new_zeros(())
        for k, (a_, b_) in enumerate(self.align):
            ia, ib = names.index(a_), names.index(b_)
            loss_align = loss_align + self._align_loss(k, e[a_], e[b_], av[:, ia] & av[:, ib] & ~deg[a_] & ~deg[b_])
        loss_align = loss_align / max(len(self.align), 1)
        # reliability geometry
        looks_clean = {}
        loss_rel = logits.new_zeros(()); nrel = 0
        for i, n in enumerate(names):
            if n not in u: continue
            with torch.no_grad():
                r_, a_, lr, ang = self._signals(i, u[n].detach())
                if self.training: self._update_stats(i, lr, ang, u[n].detach(), y, av[:, i] & ~deg[n])
                looks_clean[n] = (lr - self.logr_mu[i]).abs() / self.logr_var[i].sqrt() < self.geom_thr
            if clean is not None and n in clean and bool(deg[n].any()):
                uc = self.rel_embed(n, self._encode_clean(n, clean[n][deg[n]]))
                loss_rel = loss_rel + self._rel_loss(u[n][deg[n]], uc, severity[n][deg[n]].float()); nrel += 1
        loss_rel = loss_rel / max(nrel, 1)
        loss_kd = logits.new_zeros(())
        for tch, stu in self.distill:
            it, is_ = names.index(tch), names.index(stu)
            ok = av[:, it] & av[:, is_] & looks_clean.get(tch, torch.ones(B, dtype=torch.bool, device=dev))
            if bool(ok.any()): loss_kd = loss_kd + self._kd(z[stu][ok], z[tch][ok])
        if self.training: self.steps += 1
        aux = (self.w['align'] * loss_align + self.w['branch'] * loss_branch + self.w['proto'] * loss_proto
               + self.w['kd'] * loss_kd + self.w['rel'] * loss_rel)
        out['loss'] = 0.75 * loss_cls + 0.25 * aux
        out['parts'] = {'cls': float(loss_cls), 'align': float(loss_align), 'branch': float(loss_branch), 'proto': float(loss_proto),
                        'kd': float(loss_kd), 'rel': float(loss_rel)}
        return logits, out
