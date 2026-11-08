import argparse, importlib, json, math, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from sklearn.metrics import roc_auc_score
from .model import HyRAF
from .corruptions import degrade


def collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


def macro_auroc(task, y, p):
    if task == 'multiclass':
        per = [roc_auc_score(y == c, p[:, c]) for c in range(p.shape[1]) if 0 < (y == c).sum() < len(y)]
    else:
        per = [roc_auc_score(y[:, c], p[:, c]) for c in range(p.shape[1]) if 0 < y[:, c].sum() < len(y)]
    return float(np.mean(per)), [float(v) for v in per]


@torch.no_grad()
def evaluate(model, dl, dev, task):
    model.eval(); P, Y, W = [], [], []
    for b in dl:
        if b is None: continue
        x, av, y = b
        logits, out = model({k: v.to(dev, non_blocking=True) for k, v in x.items()}, {k: v.to(dev) for k, v in av.items()})
        P.append((torch.softmax(logits, -1) if task == 'multiclass' else torch.sigmoid(logits)).float().cpu().numpy())
        Y.append(y.numpy()); W.append(out['weights'].cpu().numpy())
    P, Y, W = np.concatenate(P), np.concatenate(Y), np.concatenate(W)
    auc, per = macro_auroc(task, Y, P)
    return auc, per, P, Y, W


def run_fold(ds_name, fold, args, dev):
    mod = importlib.import_module(f'hyraf.data.{ds_name}')
    paths = {'data_root': args.data_root}
    if ds_name == 'mimic': paths.update(ecg_root=args.ecg_root, cxr_root=args.cxr_root, ecg_ckpt=args.ecg_ckpt)
    tr, va, te, spec = mod.build(fold, **paths)
    seed = args.seed + fold - 1
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    out_dir = os.path.join(args.out, ds_name, f'fold{fold}'); os.makedirs(out_dir, exist_ok=True)
    bs = spec['batch_size']; task = spec['task']
    dl_tr = DataLoader(tr, bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate, persistent_workers=True)
    dl_va = DataLoader(va, bs, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate)
    dl_te = DataLoader(te, bs, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate)

    model = HyRAF(spec['modalities'], spec['num_class'], spec['align'], spec['distill'], task=spec['task'],
                  embed_dim=spec['embed_dim'], queue=spec['align_queue']).to(dev)
    if task == 'multiclass' and spec.get('class_weighted'):
        y = np.array([tr.labels[i] for i in tr.ids]); C = spec['num_class']
        cw = torch.tensor([len(y) / (C * max((y == c).sum(), 1)) for c in range(C)], dtype=torch.float32, device=dev)
    elif task == 'multilabel' and spec.get('pos_weight'):
        cw = torch.tensor(spec['pos_weight'], dtype=torch.float32, device=dev)
    else:
        cw = None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW if spec['optimizer'] == 'adamw' else torch.optim.Adam)(params, lr=spec['lr'], weight_decay=spec['weight_decay'])
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec['epochs'], eta_min=1e-6) if spec['schedule'] == 'cosine'
           else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, mode='max'))
    gen = torch.Generator().manual_seed(seed)
    epochs = 2 if args.smoke else spec['epochs']
    best, bad, skipped = -1.0, 0, 0
    log = open(os.path.join(out_dir, 'train.log'), 'a')
    for ep in range(epochs):
        model.train(); t0 = time.time(); agg = {}; n = 0
        for bi, b in enumerate(dl_tr):
            if b is None: continue
            if args.smoke and bi >= 5: break
            x, av, y = b
            x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}; av = {k: v.to(dev) for k, v in av.items()}; y = y.to(dev)
            xd, sev = degrade(x, av, spec['degrade'], gen)
            _, out = model(xd, av, y=y, clean=x, severity=sev, class_weight=cw)
            loss = out['loss']
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss): skipped += 1; continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 5.0)
            if not torch.isfinite(gn): opt.zero_grad(set_to_none=True); skipped += 1; continue
            opt.step()
            for k_, v_ in out['parts'].items(): agg[k_] = agg.get(k_, 0.0) + v_
            n += 1
        vauc, _, _, _, W = evaluate(model, dl_va, dev, task)
        if spec['schedule'] == 'cosine': sch.step()
        else: sch.step(vauc)
        msg = (f'ep {ep + 1:3d} val_auroc {vauc:.4f} ' + ' '.join(f'{k}={v / max(n, 1):.4f}' for k, v in agg.items())
               + f' w={np.round(W.mean(0), 3).tolist()} skipped={skipped} {time.time() - t0:.0f}s')
        print(f'[{ds_name} f{fold}] {msg}', flush=True); log.write(msg + '\n'); log.flush()
        if vauc > best:
            best, bad = vauc, 0; torch.save({'epoch': ep, 'val_auroc': vauc, 'state_dict': model.state_dict()}, os.path.join(out_dir, 'best.pth'))
        else:
            bad += 1
            if bad >= spec['patience']: break
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pth'), map_location=dev, weights_only=True)['state_dict'])
    auc, per, P, Y, W = evaluate(model, dl_te, dev, task)
    res = {'dataset': ds_name, 'fold': fold, 'seed': seed, 'val_auroc': best, 'test_auroc': auc, 'test_auroc_per_class': per,
           'mean_fusion_weights': W.mean(0).tolist(), 'skipped_steps': skipped}
    json.dump(res, open(os.path.join(out_dir, 'test.json'), 'w'), indent=1)
    np.savez(os.path.join(out_dir, 'test_pred.npz'), prob=P, label=Y, weights=W)
    print(f'[{ds_name} f{fold}] TEST macro-AUROC {auc:.4f}', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['amd', 'mimic', 'adni'])
    ap.add_argument('--folds', default='1,2,3,4,5')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--ecg_root', default='', help='MIMIC only: MIMIC-IV-ECG WFDB root')
    ap.add_argument('--cxr_root', default='', help='MIMIC only: MIMIC-CXR JPEG folder')
    ap.add_argument('--ecg_ckpt', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained', 'ecg_res18.pth'),
                    help='MIMIC only: pretrained ECG ResNet18 weights')
    ap.add_argument('--out', default='runs')
    ap.add_argument('--smoke', action='store_true', help='2 short epochs, for checking the installation')
    args = ap.parse_args()
    if args.dataset == 'mimic':
        for k in ('ecg_root', 'cxr_root'):
            if not getattr(args, k): ap.error(f'--{k} is required for --dataset mimic')
        if not os.path.isfile(args.ecg_ckpt): ap.error(f'pretrained ECG weights not found: {args.ecg_ckpt}')
    dev = torch.device(f'cuda:{args.gpu}')
    res = [run_fold(args.dataset, int(k), args, dev) for k in args.folds.split(',')]
    s = {'folds': [r['fold'] for r in res], 'test_auroc': [r['test_auroc'] for r in res],
         'mean': float(np.mean([r['test_auroc'] for r in res])), 'std': float(np.std([r['test_auroc'] for r in res]))}
    json.dump(s, open(os.path.join(args.out, args.dataset, 'summary.json'), 'w'), indent=1)
    print(json.dumps(s), flush=True)


if __name__ == '__main__':
    main()
