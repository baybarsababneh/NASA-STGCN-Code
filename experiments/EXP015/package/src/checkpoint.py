"""Streaming implementation of the unchanged hash-prefixed checkpoint format."""
import hashlib, os, shutil, uuid
from pathlib import Path
import torch

TAG = b'EXP013_CHECKPOINT_V1\n'

def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = path.with_name(path.name + '.tmp-payload-' + token)
    wrapped = path.with_name(path.name + '.tmp-' + token)
    try:
        with payload.open('wb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        h = hashlib.sha256()
        with payload.open('rb') as f:
            for block in iter(lambda: f.read(2**20), b''): h.update(block)
        with wrapped.open('wb') as dst, payload.open('rb') as src:
            dst.write(TAG + h.hexdigest().encode() + b'\n')
            shutil.copyfileobj(src, dst, 2**20); dst.flush(); os.fsync(dst.fileno())
        os.replace(wrapped, path)
    finally:
        payload.unlink(missing_ok=True); wrapped.unlink(missing_ok=True)

class PayloadView:
    def __init__(self, file, offset): self.file, self.offset = file, offset
    def read(self, n=-1): return self.file.read(n)
    def readinto(self, b): return self.file.readinto(b)
    def tell(self): return self.file.tell() - self.offset
    def seek(self, offset, whence=0):
        if whence == 0: return self.file.seek(self.offset + offset) - self.offset
        return self.file.seek(offset, whence) - self.offset

def load_torch(path):
    with Path(path).open('rb') as f:
        if f.readline() != TAG: raise RuntimeError('Checkpoint tag corruption')
        expected = f.readline().strip().decode(); offset = f.tell()
        if len(expected) != 64: raise RuntimeError('Checkpoint hash corruption')
        h = hashlib.sha256()
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
        if h.hexdigest() != expected: raise RuntimeError('Checkpoint payload corruption')
        f.seek(offset)
        return torch.load(PayloadView(f, offset), map_location='cpu', weights_only=True)
