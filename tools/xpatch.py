#!/usr/bin/env python3
"""XOR-transport patch: patch = <codec>(A XOR B); apply = A XOR <codec>-1(patch).

Byte-level and format-blind: no GGUF parsing, no ML libraries, no quantizer.
For GGUF variants of the same layout this beats range-based patches by ~3x
(identical bytes become 0x00 and compress to nothing; the changed-byte XORs
compress somewhat too). Works for equal or differing sizes.

  make :  xpatch.py make  --base A --target B --patch P [--codec xz|xz-cli|zstd]
                          [--manifest M.json] [--provenance P.json]
  apply:  xpatch.py apply --base A --patch P --manifest M.json --out OUT [--codec OVERRIDE]

Requirements:
  - Python 3.8+ — stdlib only for codec "xz" (numpy optional: used if importable,
    disable with XPATCH_NO_NUMPY=1)
  - codec "xz-cli": requires the `xz` binary (independent decoder path — cross-checks
    the stdlib lzma decoder; both decode each other's files byte-identically)
  - codec "zstd": requires the `zstd` binary

Manifest (schema "model-patch/1"): base/target sha256 + sizes, codec, compressor
description, patch sha256/bytes, xor_stream_bytes, created timestamp, and a
free-form `provenance` dict supplied by the producer (model name/url/revision,
quantizer build/commit, tool versions, ...). `apply` verifies base + patch hashes
before touching the output and the target hash after — fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import operator
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

if os.environ.get("XPATCH_NO_NUMPY"):
    np = None
else:
    try:
        import numpy as np
    except ImportError:  # optional
        np = None

CHUNK = 1 << 24
XZ_PRESET = 9 | lzma.PRESET_EXTREME
SCHEMA = "model-patch/1"

CLI_CODECS = {
    "xz-cli": {"make": ["xz", "-9e", "-T8", "-c"], "apply": ["xz", "-d", "-c"]},
    "zstd": {"make": ["zstd", "-19", "-T8", "--long=27", "-c"], "apply": ["zstd", "-d", "-c"]},
}
CODEC_CHOICES = ["xz", "xz-cli", "zstd"]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if np is not None:
        return (np.frombuffer(a, dtype=np.uint8) ^ np.frombuffer(b, dtype=np.uint8)).tobytes()
    return bytes(map(operator.xor, a, b))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _cli_version(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        return out[0] if out else "unknown"
    except Exception:
        return "unavailable"


def compressor_desc(codec: str) -> str:
    if codec == "xz":
        return f"python-lzma preset=9|EXTREME (python {platform.python_version()})"
    if codec == "xz-cli":
        return _cli_version(["xz", "--version"])
    return _cli_version(["zstd", "--version"])


def open_make_stream(patch_path: str, codec: str):
    """Returns (write_stream, cleanup_callable)."""
    if codec == "xz":
        return lzma.open(patch_path, "wb", preset=XZ_PRESET), (lambda: None)
    fout = open(patch_path, "wb")
    proc = subprocess.Popen(CLI_CODECS[codec]["make"], stdin=subprocess.PIPE, stdout=fout)

    def cleanup() -> None:
        proc.stdin.close()
        rc = proc.wait()
        fout.close()
        if rc != 0:
            sys.exit(f"FATAL: {codec} compressor exited with {rc}")

    return proc.stdin, cleanup


def open_apply_stream(patch_path: str, codec: str):
    """Returns (read_stream, cleanup_callable)."""
    if codec == "xz":
        return lzma.open(patch_path, "rb"), (lambda: None)
    proc = subprocess.Popen(CLI_CODECS[codec]["apply"] + [patch_path], stdout=subprocess.PIPE)

    def cleanup() -> None:
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            sys.exit(f"FATAL: {codec} decompressor exited with {rc}")

    return proc.stdout, cleanup


def cmd_make(args) -> None:
    sa, sb = os.path.getsize(args.base), os.path.getsize(args.target)
    common = min(sa, sb)
    provenance = {}
    if args.provenance:
        with open(args.provenance) as f:
            provenance = json.load(f)
    stream, cleanup = open_make_stream(args.patch, args.codec)
    try:
        with open(args.base, "rb") as fa, open(args.target, "rb") as ft:
            off = 0
            while off < common:
                n = min(CHUNK, common - off)
                stream.write(xor_bytes(fa.read(n), ft.read(n)))
                off += n
            off = sa
            while off < sb:  # target tail (xor with nothing = identity)
                n = min(CHUNK, sb - off)
                stream.write(ft.read(n))
                off += n
    finally:
        stream.close()
        cleanup()
    manifest = {
        "schema": SCHEMA,
        "base": os.path.abspath(args.base),
        "target": os.path.abspath(args.target),
        "base_sha256": sha256_file(args.base),
        "target_sha256": sha256_file(args.target),
        "base_size": sa,
        "target_size": sb,
        "codec": args.codec,
        "compressor": compressor_desc(args.codec),
        "xor_stream_bytes": max(sa, sb),
        "patch": os.path.abspath(args.patch),
        "patch_sha256": sha256_file(args.patch),
        "patch_bytes": os.path.getsize(args.patch),
        "patch_pct_of_base": round(100.0 * os.path.getsize(args.patch) / sa, 4),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance,
    }
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def cmd_apply(args) -> None:
    with open(args.manifest) as f:
        manifest = json.load(f)
    codec = args.codec or manifest["codec"]
    if args.codec and args.codec != manifest["codec"]:
        print(f"codec override: {manifest['codec']} -> {codec} (decoder cross-check)")
    base_sha = sha256_file(args.base)
    if base_sha != manifest["base_sha256"]:
        sys.exit(f"FATAL: base hash mismatch\n  want {manifest['base_sha256']}\n  have {base_sha}")
    if os.path.getsize(args.base) != manifest["base_size"]:
        sys.exit("FATAL: base size mismatch")
    print(f"base sha256 OK: {base_sha[:16]}...")
    have = sha256_file(args.patch)
    if have != manifest["patch_sha256"]:
        sys.exit(f"FATAL: patch hash mismatch\n  want {manifest['patch_sha256']}\n  have {have}")
    print(f"patch sha256 OK: {have[:16]}...")
    prov = manifest.get("provenance") or {}
    base_model = (prov.get("base_model") or {}).get("name")
    quantizer = prov.get("quantizer") or {}
    if base_model or quantizer:
        print(f"provenance: base={base_model or '?'} "
              f"quantizer={quantizer.get('tool', '?')} {quantizer.get('build', '')} "
              f"{quantizer.get('quant', '')}".rstrip())
    target_sha = manifest["target_sha256"]
    target_size = manifest["target_size"]
    sa = os.path.getsize(args.base)
    stream, cleanup = open_apply_stream(args.patch, codec)
    try:
        with open(args.base, "rb") as fb, open(args.out, "wb") as fo:
            off = 0
            while True:
                blk = stream.read(CHUNK)
                if not blk:
                    break
                if off < sa:
                    n = min(len(blk), sa - off)
                    fo.write(xor_bytes(blk[:n], fb.read(n)))
                    if n < len(blk):
                        fo.write(blk[n:])
                    off += n
                else:
                    fo.write(blk)
            fo.truncate(target_size)
    finally:
        stream.close()
        cleanup()
    out_sha = sha256_file(args.out)
    ok = out_sha == target_sha
    print(f"target sha256 {'MATCH' if ok else 'MISMATCH'}: {out_sha}")
    if not ok:
        sys.exit("FATAL: patched file does not match target")
    print(f"PASS: {args.base} + {args.patch} -> {args.out} == target ({target_sha[:16]}...)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--base", required=True)
    m.add_argument("--target", required=True)
    m.add_argument("--patch", required=True)
    m.add_argument("--manifest", default=None)
    m.add_argument("--provenance", default=None,
                   help="JSON file with provenance metadata merged into the manifest")
    m.add_argument("--codec", choices=CODEC_CHOICES, default="xz",
                   help="xz: stdlib lzma, no external binaries; xz-cli: external xz binary "
                        "(independent decoder); zstd: fastest decompress, ~2.6%% larger")
    m.set_defaults(fn=cmd_make)
    a = sub.add_parser("apply")
    a.add_argument("--base", required=True)
    a.add_argument("--patch", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--codec", choices=CODEC_CHOICES, default=None,
                   help="override the manifest's codec (e.g. xz-cli to cross-check stdlib lzma)")
    a.set_defaults(fn=cmd_apply)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
