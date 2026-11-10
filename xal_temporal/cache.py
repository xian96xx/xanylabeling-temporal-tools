"""Bounded, in-process memoization of exactly matching SAM video chunks."""
from collections import OrderedDict
import hashlib
import json
import numpy as np


class TrackingChunkCache:
    def __init__(self, limit=128):
        self.limit = limit
        self.entries = OrderedDict()

    @staticmethod
    def fingerprints(paths):
        result = []
        for path in paths:
            before = path.stat()
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('图片在计算期间发生变化，请重新生成预览。')
            result.append((str(path.resolve()), digest))
        return tuple(result)

    @staticmethod
    def key(fingerprints, box, carry, rotor, margins):
        if carry is None:
            seed = ('box', tuple(float(x) for x in box))
        else:
            packed = np.packbits(np.asarray(carry, dtype=bool), bitorder='little')
            seed = ('mask', tuple(carry.shape), hashlib.sha256(packed.tobytes()).hexdigest())
        # Margins affect the returned boxes even when the underlying masks match.
        margin_key = json.dumps(margins, sort_keys=True, separators=(',', ':')) if rotor else None
        return (fingerprints, seed, bool(rotor), margin_key)

    def get(self, key):
        value = self.entries.get(key)
        if value is None: return None
        self.entries.move_to_end(key)
        boxes, evidence, mask, shape = value
        carry = np.unpackbits(mask, count=int(np.prod(shape)), bitorder='little').astype(bool).reshape(shape)
        return ([None if b is None else list(b) for b in boxes],
                [None if b is None else list(b) for b in evidence], carry)

    def put(self, key, boxes, evidence, carry):
        if carry is None: return
        shape = tuple(carry.shape)
        packed = np.packbits(np.asarray(carry, dtype=bool).reshape(-1), bitorder='little')
        self.entries[key] = (tuple(None if b is None else tuple(b) for b in boxes),
                             tuple(None if b is None else tuple(b) for b in evidence),
                             packed, shape)
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit: self.entries.popitem(last=False)
