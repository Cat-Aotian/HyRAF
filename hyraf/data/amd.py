import os
import numpy as np
import torch
import nibabel as nib
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from . import load_folds
from .. import encoders as E


class AMD(Dataset):
    def __init__(self, root, ids, labels, n_slices=64, oct_size=200):
        self.root, self.ids, self.labels, self.n, self.size = root, list(ids), labels, n_slices, oct_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]; d = os.path.join(self.root, 'Image', sid)
        f = np.asarray(Image.open(os.path.join(d, f'data_{sid}_fundus.png')).convert('L'), dtype=np.float32)
        if f.shape[0] != 448: f = np.asarray(Image.fromarray(f.astype(np.uint8)).resize((448, 448)), dtype=np.float32)
        v = np.transpose(nib.load(os.path.join(d, f'processed_data_{sid}.nii')).get_fdata().astype(np.float32), (2, 0, 1))
        if v.shape[0] > self.n: v = v[np.linspace(0, v.shape[0] - 1, self.n, dtype=int)]
        if v.shape[1] != self.size: v = ndimage.zoom(v, (1, self.size / v.shape[1], self.size / v.shape[1]), order=1)
        x = {'fundus': torch.from_numpy(f / 255.0)[None].float(), 'oct': torch.from_numpy(v / 255.0).float()}
        avail = {'fundus': torch.tensor(True), 'oct': torch.tensor(True)}
        return x, avail, torch.tensor(self.labels[sid], dtype=torch.long)


def build(fold, data_root):
    labels, f = load_folds('amd', fold)
    mk = lambda ids: AMD(data_root, ids, labels)
    spec = dict(
        modalities={'fundus': (E.FundusEncoder(), 512, True), 'oct': (E.OCTEncoder(), 512, True)},
        num_class=4, task='multiclass', embed_dim=512,
        align_queue=1024,
        align=[('fundus', 'oct')], distill=[('oct', 'fundus')],
        degrade=[(0.25, ('fundus', 'oct'))],
        optimizer='adam', lr=1e-4, weight_decay=1e-4, schedule='plateau', batch_size=8, epochs=100, patience=20,
        class_weighted=True)
    return mk(f['train']), mk(f['val']), mk(f['test']), spec
