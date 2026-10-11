#!/usr/bin/env python3
"""Compare two GGUF files structurally: metadata KV sizes, data offset, tensor maps.

Usage: gguf_cmp.py A.gguf B.gguf [A-map.json B-map.json]

Requires gguf-py on PYTHONPATH (e.g. PYTHONPATH=.tools/llama-conv/gguf-py).
Writes tensor maps in the format attribute.py consumes:
    {"tensors": [{"name","offset","n_bytes","type","shape"}, ...]}
"""
from __future__ import annotations

import json
import sys

import numpy as np
from gguf import GGUFReader


def nbytes_of(t) -> int:
    try:
        return int(t.n_bytes)
    except Exception:
        return int(np.prod(t.shape)) * 0  # unknown; caller may still compare offsets


def main() -> None:
    pa, pb = sys.argv[1], sys.argv[2]
    ra, rb = GGUFReader(pa), GGUFReader(pb)

    print(f"== {pa}")
    print(f"== {pb}")
    print()
    print("data section offsets (tensor data start):")
    print(f"  A: {ra.data_offset}")
    print(f"  B: {rb.data_offset}")
    print(f"  delta: {rb.data_offset - ra.data_offset}")

    fa, fb = ra.fields, rb.fields
    print()
    print(f"metadata fields: A={len(fa)} B={len(fb)}")

    def fsize(f):
        return sum(int(p.nbytes) for p in f.parts)

    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    if only_a:
        print("fields only in A:", only_a)
    if only_b:
        print("fields only in B:", only_b)
    sized = [(n, fsize(fa[n]), fsize(fb[n])) for n in sorted(set(fa) & set(fb))
             if fsize(fa[n]) != fsize(fb[n])]
    print(f"field value-size diffs: {len(sized)}")
    for n, x, y in sized[:60]:
        print(f"  {n}: A={x} B={y} d={y - x}")

    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    print()
    print(f"tensors: A={len(ta)} B={len(tb)}")
    only_ta = sorted(set(ta) - set(tb))
    only_tb = sorted(set(tb) - set(ta))
    if only_ta:
        print(f"tensors only in A ({len(only_ta)}):", only_ta[:20], "..." if len(only_ta) > 20 else "")
    if only_tb:
        print(f"tensors only in B ({len(only_tb)}):", only_tb[:20], "..." if len(only_tb) > 20 else "")

    tdiff = [(n, ta[n].tensor_type, tb[n].tensor_type, tuple(int(v) for v in ta[n].shape),
              tuple(int(v) for v in tb[n].shape))
             for n in sorted(set(ta) & set(tb))
             if ta[n].tensor_type != tb[n].tensor_type or tuple(ta[n].shape) != tuple(tb[n].shape)]
    print(f"tensor type/shape diffs: {len(tdiff)}")
    for d in tdiff[:60]:
        print("  ~", d)

    odiff = [(n, int(ta[n].data_offset), int(tb[n].data_offset))
             for n in sorted(set(ta) & set(tb))
             if int(ta[n].data_offset) != int(tb[n].data_offset)]
    print(f"tensors with differing absolute data_offset: {len(odiff)}")
    for d in odiff[:20]:
        print("  off", d)

    if len(sys.argv) > 3:
        def mapof(tensors):
            return {"tensors": [
                {"name": t.name,
                 "offset": int(t.data_offset),
                 "n_bytes": nbytes_of(t),
                 "type": str(t.tensor_type),
                 "shape": [int(v) for v in t.shape]}
                for t in tensors]}
        with open(sys.argv[3], "w") as f:
            json.dump(mapof(ra.tensors), f, indent=1)
        with open(sys.argv[4], "w") as f:
            json.dump(mapof(rb.tensors), f, indent=1)
        print("maps written:", sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
