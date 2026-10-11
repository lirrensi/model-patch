#!/usr/bin/env python3
"""Rewrite a GGUF with its metadata KV section copied verbatim from a reference GGUF.

Keeps the target's tensors, writes a new file whose metadata KV section matches
the reference's keys/values/types/order — so tensor data offsets align when the
tensor sets match (same names, shapes, types, order).

Usage: gguf_renorm.py <target.gguf> <reference.gguf> <out.gguf>

Requires gguf-py on PYTHONPATH.
"""
from __future__ import annotations

import sys

from gguf import GGUFReader, GGUFWriter, GGUFValueType


def main() -> None:
    target_p, ref_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    ref = GGUFReader(ref_p)
    tgt = GGUFReader(target_p)

    arch = str(ref.fields["general.architecture"].contents())
    w = GGUFWriter(out_p, arch)

    for name, f in ref.fields.items():
        if name.startswith("GGUF."):
            # reader pseudo-fields for header entries (GGUF.version, GGUF.tensor_count,
            # GGUF.kv_count) — never written as KV pairs
            continue
        if name == "general.architecture":
            # GGUFWriter(arch=...) already added it as the first KV
            continue
        val = f.contents()
        vtype = f.types[0]
        sub = None
        if vtype == GGUFValueType.ARRAY:
            sub = f.types[1] if len(f.types) > 1 else GGUFValueType.STRING
        w.add_key_value(name, val, vtype, sub_type=sub)

    for t in tgt.tensors:
        # GGUFReader.shape is in GGUF/ne order; GGUFWriter reverses raw_shape when
        # writing ne[] — so pass numpy order.
        w.add_tensor(t.name, t.data, raw_shape=list(t.shape)[::-1], raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"rewrote {target_p} with metadata from {ref_p} -> {out_p}")


if __name__ == "__main__":
    main()
