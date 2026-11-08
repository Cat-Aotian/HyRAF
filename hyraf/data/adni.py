import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from . import load_folds
from .. import encoders as E


class ADNI(Dataset):
    def __init__(self, root, ids, labels, train):
        self.root, self.ids, self.labels, self.train = root, list(ids), labels, train
        self.cli = np.load(os.path.join(root, 'clinical/clinical_features.npy')).astype(np.float32)
        self.cli_idx = {p.strip(): i for i, p in enumerate(open(os.path.join(root, 'clinical/clinical_ptids.txt')))}
        self.bio = np.load(os.path.join(root, 'biospecimen/biospecimen_features.npy')).astype(np.float32)
        self.bio_idx = {p.strip(): i for i, p in enumerate(open(os.path.join(root, 'biospecimen/biospecimen_ptids.txt')))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        p = self.ids[i]
        img = np.load(os.path.join(self.root, 'image/processed_3d', f'{p}_brain.npy')).astype(np.float32)[None]
        if self.train:
            if np.random.rand() < 0.5: img = img[:, ::-1].copy()
            if np.random.rand() < 0.5: img = img[:, :, ::-1].copy()
            if np.random.rand() < 0.5: img = img + np.random.randn(*img.shape).astype(np.float32) * 0.005
        has_c, has_b = p in self.cli_idx, p in self.bio_idx
        cli = self.cli[self.cli_idx[p]] if has_c else np.zeros(self.cli.shape[1], np.float32)
        bio = self.bio[self.bio_idx[p]] if has_b else np.zeros(self.bio.shape[1], np.float32)
        x = {'mri': torch.from_numpy(img), 'clinical': torch.from_numpy(cli), 'biospecimen': torch.from_numpy(bio)}
        avail = {'mri': torch.tensor(True), 'clinical': torch.tensor(has_c), 'biospecimen': torch.tensor(has_b)}
        return x, avail, torch.tensor(self.labels[p], dtype=torch.long)


def build(fold, data_root):
    labels, f = load_folds('adni', fold)
    spec = dict(
        modalities={'mri': (E.MRIEncoder(), 512, True), 'clinical': (E.ClinicalEncoder(), 256, True),
                    'biospecimen': (E.BiospecimenEncoder(), 64, False)},
        num_class=3, task='multiclass', embed_dim=512,
        align_queue=1024,
        align=[('mri', 'clinical')], distill=[],
        degrade=[(0.25, ('mri',)), (0.15, ('clinical',))],
        optimizer='adamw', lr=1e-4, weight_decay=1e-5, schedule='cosine', batch_size=16, epochs=150, patience=20,
        class_weighted=False)
    return (ADNI(data_root, f['train'], labels, True), ADNI(data_root, f['val'], labels, False),
            ADNI(data_root, f['test'], labels, False), spec)
