import os
import numpy as np
import torch
import wfdb
from PIL import Image
from scipy import signal as S
import torchvision.transforms as T
from torch.utils.data import Dataset
from . import load_folds
from .. import encoders as E

LABEL_COLS = [11, 12, 13, 14]
_B, _A = S.butter(5, 1 / (0.5 * 400), btype='high')


def _read_rows(csv_dir):
    rows = {}
    for split in ('train', 'val', 'test'):
        lines = open(os.path.join(csv_dir, f'with_nonan_label_PA_{split}.csv')).read().splitlines()[1:]
        for line in lines:
            m = line.split(',')
            rows[m[1]] = dict(ecg=m[6].strip(), dicom=m[7].strip(), y=[1 if m[i] == '1' else 0 for i in LABEL_COLS])
    return rows


class MIMIC(Dataset):
    def __init__(self, rows, ids, train, ecg_root, cxr_root):
        self.rows, self.ids, self.ecg_root, self.cxr_root = rows, [i for i in ids if i in rows], ecg_root, cxr_root
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = T.Compose(([T.Resize(256), T.RandomHorizontalFlip(), T.RandomAffine(45, translate=(0.15, 0.15), scale=(.85, 1.15))]
                             if train else [T.Resize(256)]) + [T.CenterCrop(224), T.ToTensor(), norm])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        r = self.rows[self.ids[i]]
        try:
            sig = wfdb.rdrecord(os.path.join(self.ecg_root, r['ecg'])).p_signal
            sig = np.pad(S.resample(sig, 4000, axis=0), ((48, 48), (0, 0))).astype(np.float32)
            if np.isnan(sig).any(): return None
            sig = S.filtfilt(_B, _A, sig.T).T.astype(np.float32)            # 1 Hz high-pass, [4096, 12]
            img = self.tf(Image.open(os.path.join(self.cxr_root, f"{r['dicom']}.jpg")).convert('RGB'))
        except (OSError, ValueError):
            return None
        x = {'ecg': torch.from_numpy(np.ascontiguousarray(sig)), 'cxr': img}
        return x, {'ecg': torch.tensor(True), 'cxr': torch.tensor(True)}, torch.tensor(r['y'], dtype=torch.long)


def build(fold, data_root, ecg_root, cxr_root, ecg_ckpt):
    _, f = load_folds('mimic', fold); rows = _read_rows(data_root)
    mk = lambda ids, tr: MIMIC(rows, ids, tr, ecg_root, cxr_root)
    spec = dict(
        modalities={'ecg': (E.ECGEncoder(ecg_ckpt), 512, True), 'cxr': (E.CXREncoder(), 256, True)},
        num_class=4, task='multilabel', embed_dim=1024,
        align_queue=0,
        align=[('ecg', 'cxr')], distill=[],
        degrade=[(0.25, ('ecg', 'cxr'))],
        optimizer='adam', lr=1e-4, weight_decay=1e-5, schedule='cosine', batch_size=64, epochs=40, patience=8,
        pos_weight=[2.88, 2.34, 2.38, 2.57])
    return mk(f['train'], True), mk(f['val'], False), mk(f['test'], False), spec
