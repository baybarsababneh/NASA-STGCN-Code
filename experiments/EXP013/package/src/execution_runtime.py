"""Logging and synchronous backup hooks; never part of model configuration."""
import json, time, contextlib
from pathlib import Path
import torch

ACTIVE = None

@contextlib.contextmanager
def operation(name, **details):
    if ACTIVE is not None and hasattr(ACTIVE, 'operation'):
        with ACTIVE.operation(name, **details): yield
    else: yield

def gradients(model):
    if ACTIVE is not None and hasattr(ACTIVE, 'verify_gradients'):
        ACTIVE.verify_gradients(model)

def due():
    return ACTIVE is not None and ACTIVE.checkpoint_due()

def checkpoint(work, reason, force=False):
    if ACTIVE is not None: ACTIVE.backup(reason, force=force)

def progress(work, job, phase, epoch, batch, batches, start, epochs,
             epoch_elapsed=0., force=False, **extra):
    if ACTIVE is not None:
        ACTIVE.progress(job, phase, epoch, batch, batches, start, epochs,
                        epoch_elapsed=epoch_elapsed, force=force, **extra)
    elif force or batch == 1 or batch == batches or batch % 20 == 0:
        print(json.dumps({'job':job, 'phase':phase, 'epoch':epoch,
            'batch':batch, 'batches':batches,
            'elapsed_seconds':round(time.monotonic()-start+epoch_elapsed, 2)},
            sort_keys=True), flush=True)
