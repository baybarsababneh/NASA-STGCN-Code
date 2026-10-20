"""Engineering-only vectorised collation and serialisable original sampler order."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class Indices(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i

def epoch_plan(loader):
    # Use the PINNED DataLoader, including its base-seed draw and final sampler
    # exhaustion. The dedicated generator is used only for sampling, never for
    # model/dropout/injection draws. Materialising indices changes no such draw.
    index_loader = DataLoader(Indices(len(loader.dataset)),
        batch_sampler=loader.batch_sampler, generator=loader.generator,
        num_workers=0)
    return [batch.tolist() for batch in index_loader]

def gather(dataset, indices):
    """Exactly the old default_collate bytes, without per-window Tensor creation."""
    B = len(indices)
    groups = defaultdict(list)
    entities = np.empty(B, dtype=np.int64)
    target = np.empty(B, dtype=np.float32)
    for pos, i in enumerate(indices):
        ch, t, e = dataset.rows[i]
        groups[ch].append((pos, t))
        entities[pos] = e
        target[pos] = dataset.raw[ch][t, 0]
    nf = dataset.raw[dataset.rows[indices[0]][0]].shape[1]
    if dataset.kind in ['graph', 'removed', 'fusion']:
        x = np.empty((B, 65, nf), dtype=np.float32)
        for ch, rows in groups.items():
            pos, ends = np.asarray(rows, dtype=np.int64).T
            x[pos] = dataset.raw[ch][ends[:, None] + np.arange(-64, 1)[None]]
        x[:, -1, 1:] = 0
        return (torch.from_numpy(x), torch.from_numpy(entities),
                torch.from_numpy(target))
    if dataset.kind == 'gdn':
        x = np.empty((B, nf, 5), dtype=np.float32)
        y = np.empty((B, nf), dtype=np.float32)
        for ch, rows in groups.items():
            pos, ends = np.asarray(rows, dtype=np.int64).T
            x[pos] = dataset.raw[ch][ends[:, None] + np.arange(-5, 0)[None]].transpose(0, 2, 1)
            y[pos] = dataset.raw[ch][ends]
        return torch.from_numpy(x), torch.from_numpy(y)
    x = np.empty((B, 192, nf), dtype=np.float32)
    for ch, rows in groups.items():
        pos, ends = np.asarray(rows, dtype=np.int64).T
        x[pos] = dataset.raw[ch][ends[:, None] + np.arange(-191, 1)[None]]
    a = torch.from_numpy(x)
    return a, a  # Both old CATCH tensors have identical values; neither is mutated.

def move(item, device):
    # Preserve dtype and synchronous transfer semantics. No AMP, TF32, workers,
    # prefetch ordering or model-side arithmetic is introduced here.
    return [a.to(device) for a in item]
