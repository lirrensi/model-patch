#!/usr/bin/env python3
"""Surgically merge a PEFT LoRA adapter into a HF safetensors model, byte-clean.

Output = byte-copy of the base safetensors with ONLY the adapter's target
tensors replaced:
    W' = (W.float() + scale * (B.float() @ A.float())).to(W.dtype),  scale = alpha / r
Everything else (MTP head, F32 smalls, header/offsets) stays byte-identical.
Sidecar files (config.json, tokenizer, README, ...) are copied from the base dir.

Usage:
    python merge_adapter_clean.py --base Qwen3.5-2B --adapter H/Qwen-2B-A --out Qwen3.5-2B-A-Clean
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def find_safetensors(model_dir: Path) -> Path:
    cands = sorted({p for p in model_dir.glob("*.safetensors") if p.is_file()}
                   | {p for p in model_dir.glob("*.safetensors-*") if p.is_file()})
    if not cands:
        raise SystemExit(f"no .safetensors found in {model_dir}")
    return cands[0]


def read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return 8 + n, hdr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="original HF model dir")
    ap.add_argument("--adapter", required=True,
                    help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="output model dir")
    args = ap.parse_args()

    torch.set_num_threads(1)

    base_dir, adir, out_dir = Path(args.base), Path(args.adapter), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_st = find_safetensors(base_dir)
    data_start, hdr = read_safetensors_header(base_st)

    with open(adir / "adapter_config.json") as f:
        acfg = json.load(f)
    r = int(acfg["r"])
    alpha = float(acfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if acfg.get("use_rslora") else alpha / r
    print(f"adapter: r={r} alpha={alpha} scale={scale} targets={acfg['target_modules']}")

    # map adapter tensors -> base tensor names
    targets: dict[str, dict[str, str]] = {}
    with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as af:
        for name in af.keys():
            if name.endswith(".lora_A.weight"):
                base_name, side = name[: -len(".lora_A.weight")], "A"
            elif name.endswith(".lora_B.weight"):
                base_name, side = name[: -len(".lora_B.weight")], "B"
            else:
                raise SystemExit(f"unexpected adapter tensor: {name}")
            if base_name.startswith("base_model.model."):
                base_name = base_name[len("base_model.model."):]
            targets.setdefault(base_name + ".weight", {})[side] = name

    print(f"adapter targets: {len(targets)} tensors")

    out_st = out_dir / "model.safetensors"
    print(f"copying {base_st} -> {out_st}")
    shutil.copyfile(base_st, out_st)
    for p in base_dir.iterdir():
        if p.is_file() and ".safetensors" not in p.name:
            shutil.copyfile(p, out_dir / p.name)

    merged = 0
    with safe_open(str(base_st), framework="pt") as bf, \
         safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as af, \
         open(out_st, "r+b") as fo:
        for base_name, pair in sorted(targets.items()):
            if "A" not in pair or "B" not in pair:
                raise SystemExit(f"incomplete LoRA pair for {base_name}: {pair}")
            if base_name not in hdr:
                raise SystemExit(f"base tensor missing: {base_name}")
            W = bf.get_tensor(base_name)
            A = af.get_tensor(pair["A"])
            B = af.get_tensor(pair["B"])
            ok = (A.ndim == 2 and B.ndim == 2
                  and A.shape[0] == r and B.shape[1] == r
                  and A.shape[1] == W.shape[1] and B.shape[0] == W.shape[0])
            if not ok:
                raise SystemExit(
                    f"shape mismatch for {base_name}: W={tuple(W.shape)} "
                    f"A={tuple(A.shape)} B={tuple(B.shape)}")
            delta = scale * (B.float() @ A.float())
            Wn = (W.float() + delta).to(W.dtype)
            blob = Wn.contiguous().view(torch.uint8).numpy().tobytes()
            off0, off1 = hdr[base_name]["data_offsets"]
            if len(blob) != off1 - off0:
                raise SystemExit(f"size mismatch for {base_name}: {len(blob)} != {off1 - off0}")
            fo.seek(data_start + off0)
            fo.write(blob)
            merged += 1
            print(f"  merged {base_name:64s} |delta|={float(delta.norm()):.4f}")

    print(f"merged {merged} tensors -> {out_st}")
    print("sidecars copied; header preserved byte-identical (same shapes/dtypes/offsets)")


if __name__ == "__main__":
    main()
