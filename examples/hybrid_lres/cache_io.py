"""Strict JSON/NPZ reader for frozen numerical results, not model checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

SCHEMA = 'hybrid-lres-numerical-cache-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def safe_child(root, relative):
    item = PurePosixPath(relative)
    if item.is_absolute() or '..' in item.parts or '\\' in relative:
        raise ValueError('Bundle paths must be relative and confined')
    path = root.joinpath(*item.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Bundle path escapes root')
    return path


class Bundle:
    def __init__(self, folder, expected_manifest=None):
        self.root = Path(folder).resolve()
        manifest_path = self.root / 'manifest.json'
        manifest_bytes = manifest_path.read_bytes()
        self.reference_verified = expected_manifest is not None
        if expected_manifest is not None and hashlib.sha256(manifest_bytes).hexdigest() != sha(expected_manifest):
            raise ValueError('Bundle differs from the shipped tracked reference manifest')
        self.manifest = json.loads(manifest_bytes)
        if self.manifest.get('schema') != SCHEMA:
            raise ValueError('Unsupported numerical cache schema')
        if self.manifest.get('scope') != 'frozen-result/figure replay; no refitting or event-level reproduction':
            raise ValueError('Missing explicit reproduction scope')
        for name, record in self.manifest['files'].items():
            if sha(safe_child(self.root, name)) != record['sha256']:
                raise ValueError('Bundle SHA256 mismatch: ' + name)

    def npz(self, name):
        record = self.manifest['datasets'][name]
        path = safe_child(self.root, record['file'])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if set(arrays) != set(record['arrays']):
            raise ValueError('Array membership differs: ' + name)
        for key, array in arrays.items():
            if array.dtype.hasobject or array_digest(array) != record['arrays'][key]:
                raise ValueError('Numeric array identity differs: ' + name + '/' + key)
        return arrays

    def json(self, name):
        return json.loads(safe_child(self.root, self.manifest['datasets'][name]['file']).read_text())

    def verify_external(self, path):
        record = self.manifest['external_training_archive']
        if sha(path) != record['sha256']:
            raise ValueError('Full training archive SHA256 mismatch')
        with np.load(path, allow_pickle=False) as data:
            for name, identity in record['arrays'].items():
                if array_digest(data[name]) != identity:
                    raise ValueError('Full training array differs: ' + name)
        return record
