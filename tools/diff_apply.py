#!/usr/bin/env python3
"""GGUF byte-differ + patch applier (transport proof, size-aware v2).

diff:   base + target  ->  patch.bin (offset/len/bytes ranges) + manifest.json
apply:  base + patch.bin -> out.gguf, verified against embedded sha256(target)

Patch format (little-endian):
    v1: magic[8] = b"GGPATCH1", base_sha256[32], target_sha256[32], n_ranges[u32]
    v2: magic[8] = b"GGPATCH2", base_sha256[32], target_sha256[32],
        base_size[u64], target_size[u64], n_ranges[u32]
    ranges: n * ( offset[u64], length[u32], bytes[length] )

v2 supports target size != base size: the common prefix is diffed byte-wise,
a longer target's tail is carried as one range, and apply() truncates the
result to target_size (so a shorter target needs no range for its tail).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
import time

import numpy as np

MAGIC1 = b"GGPATCH1"
MAGIC2 = b"GGPATCH2"
HDR1 = struct.Struct("<8s32s32sI")    # magic, base sha, target sha, n_ranges
HDR2 = struct.Struct("<8s32s32sQQI")  # magic, base sha, target sha, base_size, target_size, n_ranges
ENT = struct.Struct("<QI")            # offset, length
CHUNK = 1 << 24                       # 16 MiB


def sha256_file(path: str) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.digest()


def find_ranges(a_path: str, b_path: str, gap: int):
    """Coalesced (start, end) byte ranges where the two files differ.

    Handles differing sizes: the common prefix is compared byte-wise; if the
    target is longer, its tail beyond the base is appended as one range.
    A shorter target needs no range — apply() truncates to target_size.
    """
    sa, sb = os.path.getsize(a_path), os.path.getsize(b_path)
    common = min(sa, sb)
    ranges: list[list[int]] = []
    cur: list[int] | None = None
    raw_changed = 0
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        ma = mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ)
        mb = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
        off = 0
        while off < common:
            n = min(CHUNK, common - off)
            a = np.frombuffer(ma, dtype=np.uint8, count=n, offset=off)
            b = np.frombuffer(mb, dtype=np.uint8, count=n, offset=off)
            d = np.flatnonzero(a != b)
            if d.size:
                raw_changed += int(d.size)
                brk = np.flatnonzero(np.diff(d) > 1)
                starts = np.concatenate(([d[0]], d[brk + 1]))
                ends = np.concatenate((d[brk], [d[-1]]))
                for s, e in zip(starts, ends):
                    s_abs = int(s) + off
                    e_abs = int(e) + off + 1
                    if cur is not None and s_abs <= cur[1] + gap:
                        cur[1] = e_abs
                    else:
                        if cur is not None:
                            ranges.append(cur)
                        cur = [s_abs, e_abs]
            off += n
        if sb > sa:
            raw_changed += sb - sa
            if cur is not None and sa <= cur[1] + gap:
                cur[1] = sb
            else:
                if cur is not None:
                    ranges.append(cur)
                cur = [sa, sb]
        if cur is not None:
            ranges.append(cur)
        # numpy views on the mmap hold exported pointers; leave closing to process exit
    return ranges, raw_changed


def cmd_diff(args) -> None:
    t0 = time.time()
    sa, sb = os.path.getsize(args.base), os.path.getsize(args.target)
    ranges, raw_changed = find_ranges(args.base, args.target, args.gap)
    base_sha = sha256_file(args.base)
    target_sha = sha256_file(args.target)
    payload = sum(e - s for s, e in ranges)
    with open(args.patch, "wb") as f:
        f.write(HDR2.pack(MAGIC2, base_sha, target_sha, sa, sb, len(ranges)))
        with open(args.target, "rb") as ft:
            for s, e in ranges:
                ft.seek(s)
                blob = ft.read(e - s)
                f.write(ENT.pack(s, e - s))
                f.write(blob)
    patch_size = os.path.getsize(args.patch)
    manifest = {
        "base": os.path.abspath(args.base),
        "target": os.path.abspath(args.target),
        "base_sha256": base_sha.hex(),
        "target_sha256": target_sha.hex(),
        "base_size": sa,
        "target_size": sb,
        "size_delta": sb - sa,
        "n_ranges": len(ranges),
        "raw_changed_bytes": raw_changed,
        "raw_changed_pct_of_base": round(100.0 * raw_changed / sa, 4),
        "patch_payload_bytes": payload,
        "patch_payload_pct_of_base": round(100.0 * payload / sa, 4),
        "patch_total_bytes": patch_size,
        "patch_total_pct_of_base": round(100.0 * patch_size / sa, 4),
        "gap_merge": args.gap,
        "seconds": round(time.time() - t0, 2),
    }
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def _read_header(f):
    magic = f.read(8)
    if magic == MAGIC2:
        _, base_sha, target_sha, base_size, target_size, n = HDR2.unpack(
            magic + f.read(HDR2.size - 8)
        )
        return base_sha, target_sha, base_size, target_size, n
    if magic == MAGIC1:
        _, base_sha, target_sha, n = HDR1.unpack(magic + f.read(HDR1.size - 8))
        return base_sha, target_sha, None, None, n
    raise SystemExit("FATAL: bad patch magic")


def cmd_apply(args) -> None:
    with open(args.patch, "rb") as f:
        base_sha, target_sha, base_size, target_size, n = _read_header(f)
        have = sha256_file(args.base)
        if have != base_sha:
            raise SystemExit(
                f"FATAL: base hash mismatch\n  want {base_sha.hex()}\n  have {have.hex()}"
            )
        print(f"base sha256 OK: {have.hex()[:16]}...")
        if base_size is not None and os.path.getsize(args.base) != base_size:
            raise SystemExit(
                f"FATAL: base size mismatch: {os.path.getsize(args.base)} != {base_size}"
            )
        shutil.copyfile(args.base, args.out)
        with open(args.out, "r+b") as fo:
            for _ in range(n):
                off, ln = ENT.unpack(f.read(ENT.size))
                blob = f.read(ln)
                fo.seek(off)
                fo.write(blob)
            if target_size is not None:
                fo.truncate(target_size)
    out_sha = sha256_file(args.out)
    ok = out_sha == target_sha
    print(f"target sha256 {'MATCH' if ok else 'MISMATCH'}: {out_sha.hex()}")
    if not ok:
        raise SystemExit("FATAL: patched file does not match target")
    print(f"PASS: {args.base} + patch -> {args.out} == target ({target_sha.hex()[:16]}...)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff")
    d.add_argument("base")
    d.add_argument("target")
    d.add_argument("--patch", required=True)
    d.add_argument("--manifest", default=None)
    d.add_argument("--gap", type=int, default=16, help="merge ranges with gaps <= N bytes")
    d.set_defaults(fn=cmd_diff)
    a = sub.add_parser("apply")
    a.add_argument("--patch", required=True)
    a.add_argument("--base", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
