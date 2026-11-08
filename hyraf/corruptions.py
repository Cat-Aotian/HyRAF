import math
import torch
import torch.nn.functional as F


def _gauss_kernel(sigma, device, max_radius):
    r = max(1, min(int(math.ceil(3 * sigma)), max_radius))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); return k / k.sum(), r


def blur(x, sigma, dims):
    shape = x.shape; nd = len(dims); y = x.reshape(-1, 1, *shape[-nd:]).float()
    for ax in range(nd):
        k, r = _gauss_kernel(sigma, x.device, shape[-nd + ax] - 1)
        view = [1, 1] + [1] * nd; view[2 + ax] = -1
        pad = [0, 0] * nd; pad[2 * (nd - 1 - ax)] = r; pad[2 * (nd - 1 - ax) + 1] = r
        conv = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
        y = conv(F.pad(y, pad, mode='reflect'), k.view(view))
    return y.reshape(shape).to(x.dtype)


def box_mask(x, ratio, dims):
    nd = len(dims); y = x.clone(); sl = [slice(None)] * (x.dim() - nd)
    for d in x.shape[-nd:]:
        L = max(1, int(round(d * ratio ** (1.0 / nd)))); o = int(torch.randint(0, max(1, d - L + 1), (1,)))
        sl.append(slice(o, o + L))
    y[tuple(sl)] = 0; return y


def noise(x, sigma):
    return x + sigma * torch.randn_like(x)


def speckle(x, sigma):
    return x * (1 + sigma * torch.randn_like(x))


def rician(x, sigma):
    return torch.sqrt((x + sigma * torch.randn_like(x)) ** 2 + (sigma * torch.randn_like(x)) ** 2)


def bias_field(x, strength):
    D, H, W = x.shape[-3:]
    z, yy, xx = torch.meshgrid(*(torch.linspace(-1, 1, n, device=x.device) for n in (D, H, W)), indexing='ij')
    return x * (1 + strength * (xx ** 2 + yy ** 2 + 0.5 * z ** 2))


def snr_mix(x, a):
    return x * (1 - a) + a * x.std() * torch.randn_like(x)


def time_mask(x, frac):
    T = x.shape[0]; L = max(1, int(T * frac)); o = int(torch.randint(0, T - L + 1, (1,)))
    y = x.clone(); y[o:o + L] = 0; return y


def lead_drop(x, k):
    y = x.clone(); y[:, torch.randperm(x.shape[1], device=x.device)[:int(k)]] = 0; return y


def lin(a, b): return lambda v, x: (v - a) / (b - a)
def log(a, b): return lambda v, x: math.log(v / a) / math.log(b / a)


# corruptions
FAMILIES = {
    # AMD (inputs in [0, 1])
    'fundus': [(lambda x, v: noise(x, v).clamp(0, 1), 0.05, 0.25, log(0.05, 1.0)),
               (lambda x, v: blur(x, v, (-2, -1)), 0.5, 5.0, lin(1, 15)),
               (lambda x, v: box_mask(x, v, (-2, -1)), 0.05, 0.5, lin(0.1, 0.9))],
    'oct': [(lambda x, v: speckle(x, v).clamp(0, 1), 0.1, 0.25, log(0.1, 2.0)),
            (lambda x, v: snr_mix(x, v).clamp(0, 1), 0.2, 0.6, lin(0, 1.2)),
            (lambda x, v: noise(x, v), 0.05, 0.3, log(0.05, 2.0)),
            (lambda x, v: box_mask(x, v, (-2, -1)), 0.05, 0.5, lin(0.1, 0.9)),
            (lambda x, v: blur(x, v, (-2, -1)), 0.5, 5.0, lin(1, 15))],
    # MIMIC (ECG [T, 12]; CXR ImageNet-normalised)
    'ecg': [(lambda x, v: noise(x, v), 0.05, 0.3, lambda v, x: math.log(v / (0.05 * float(x.std()) + 1e-8)) / math.log(20)),
            (time_mask, 0.1, 0.5, lin(0.1, 0.7)),
            (lambda x, v: lead_drop(x, math.floor(v)), 1, 7 - 1e-6, lambda v, x: (math.floor(v) - 1) / 7)],
    'cxr': [(lambda x, v: noise(x, v), 0.05, 0.3, log(0.05, 1.0)),
            (lambda x, v: blur(x, v, (-2, -1)), 1.0, 5.0, lin(0.5, 7)),
            (lambda x, v: box_mask(x, v, (-2, -1)), 0.1, 0.5, lin(0.1, 0.7))],
    # ADNI (MRI in [0, 1]; clinical features standardised)
    'mri': [(rician, 0.05, 0.25, log(0.05, 1.0)),
            (lambda x, v: blur(x, v, (-3, -2, -1)), 1.0, 5.0, lin(1, 20)),
            (lambda x, v: box_mask(x, v, (-3, -2, -1)), 0.1, 0.4, lin(0.1, 0.9)),
            (bias_field, 0.1, 0.5, lin(0.1, 1.5))],
    'clinical': [(lambda x, v: x * (torch.rand_like(x) >= v).float(), 0.1, 0.4, lin(0.1, 0.9)),
                 (lambda x, v: noise(x, v * x.std()), 0.1, 0.4, log(0.1, 2.0))],
}


@torch.no_grad()
def degrade(x, avail, groups, gen=None):
    B = next(iter(x.values())).shape[0]; dev = next(iter(x.values())).device
    y = dict(x); sev = {k: torch.full((B,), -1.0, device=dev) for k in x}
    rnd = lambda: float(torch.rand(1, generator=gen))
    for p, mods in groups:
        for b in range(B):
            if rnd() >= p: continue
            cand = [m for m in mods if bool(avail[m][b])]
            if not cand: continue
            m = cand[int(rnd() * len(cand))]; fams = FAMILIES[m]
            f, lo, hi, s = fams[int(rnd() * len(fams))]
            v = lo + (hi - lo) * rnd()
            if y[m] is x[m]: y[m] = x[m].clone()
            y[m][b] = f(x[m][b], v); sev[m][b] = min(1.0, max(0.0, s(v, x[m][b])))
    return y, sev
