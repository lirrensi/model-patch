#!/usr/bin/env python3
"""Attribute patch byte-ranges to GGUF tensors: inside touched vs outside, per-tensor.

Usage: attribute.py --patch patch.bin --tensors A-tensors.json [--pattern REGEX]

Reads both GGPATCH1 (v1) and GGPATCH2 (size-aware v2) patch files.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

HDR1 = struct.Struct("<8s32s32sI")
HDR2 = struct.Struct("<8s32s32sQQI")
ENT = struct.Struct("<QI")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--tensors", required=True)
    ap.add_argument("--pattern", default=r"\.attn_output\.weight$")
    args = ap.parse_args()

    with open(args.patch, "rb") as f:
        magic = f.read(8)
        if magic == b"GGPATCH2":
            _, _, _, base_size, target_size, n = HDR2.unpack(magic + f.read(HDR2.size - 8))
            print(f"patch v2: base_size={base_size} target_size={target_size} delta={target_size - base_size}")
        elif magic == b"GGPATCH1":
            _, _, _, n = HDR1.unpack(magic + f.read(HDR1.size - 8))
            print("patch v1")
        else:
            raise SystemExit("bad patch magic")
        ranges = []
        for _ in range(n):
            off, ln = ENT.unpack(f.read(ENT.size))
            ranges.append((off, ln))
            f.seek(ln, 1)

    tmap = json.loads(Path(args.tensors).read_text())["tensors"]
    pat = re.compile(args.pattern)
    touched = sorted(((t["name"], t["offset"], t["n_bytes"]) for t in tmap if pat.search(t["name"])),
                     key=lambda x: x[1])
    touched_total = sum(nb for _, _, nb in touched)

    changed_in = 0
    changed_out = 0
    ranges_outside = 0
    per = {name: 0 for name, _, _ in touched}
    for off, ln in ranges:
        end = off + ln
        ov_total = 0
        for name, to, tn in touched:
            o0, o1 = max(off, to), min(end, to + tn)
            if o0 < o1:
                ov_total += o1 - o0
                per[name] += o1 - o0
        changed_in += ov_total
        changed_out += ln - ov_total
        if ov_total == 0:
            ranges_outside += 1

    total_changed = changed_in + changed_out
    print(f"patch ranges:          {len(ranges)}  (fully outside touched: {ranges_outside})")
    print(f"changed bytes total:   {total_changed}")
    print(f"  inside touched:      {changed_in}  ({100.0 * changed_in / touched_total:.2f}% of touched payload)")
    print(f"  outside touched:     {changed_out}")
    print(f"touched tensors:       {len(touched)}, payload {touched_total} bytes")
    print("per tensor:")
    for name, to, tn in touched:
        c = per[name]
        print(f"  {name:36s} changed={c:>9d} / {tn:>9d}  = {100.0 * c / tn:6.2f}%")


if __name__ == "__main__":
    main()
