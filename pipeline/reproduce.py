#!/usr/bin/env python3
"""Full reproduction pipeline: HF base + LoRA adapter -> byte-exact GGUF variant + patch.

Steps (deterministic; each verified):
  0. resolve base model (local dir, or download from the Hub)
  1. surgical LoRA merge (tools/merge_adapter_clean.py)          -> merged HF dir
  2. convert base + merged to F16 GGUF (llama.cpp converter)
  3. canonicalize variant metadata to base (tools/gguf_renorm.py) + structural check (gguf_cmp)
  4. quantize both with the same llama-quantize build/flags (optional determinism gate)
  5. XOR transport patch (tools/xpatch.py make) + provenance manifest
  6. verify: apply-verify (sha256 match) + cross-decoder apply (xz CLI vs stdlib lzma)

Example:
  python pipeline/reproduce.py \
      --workdir work \
      --base-dir /models/Qwen3.5-2B \
      --adapter adapter/qwen3.5-2b-heretic-r3 \
      --llama-cpp /opt/llama.cpp/build/bin \
      --converter /opt/llama.cpp/convert_hf_to_gguf.py \
      --merge-python /venvs/heretic/bin/python \
      --pipeline-python /venvs/llama/bin/python \
      --quant Q4_K_M --gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def run(cmd, *, env=None, label=None) -> None:
    cmd = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(f"FATAL: step failed ({label or cmd[0]}), exit code {r.returncode}")
    print(f"[ok] {label or cmd[0]}  ({time.time() - t0:.1f}s)")


def capture(cmd, *, env=None) -> str:
    try:
        r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def llama_version(llama_cpp: Path) -> dict:
    exe = ".exe" if os.name == "nt" else ""
    for name in ("llama-cli", "llama-quantize"):
        p = llama_cpp / f"{name}{exe}"
        if p.exists():
            out = capture([p, "--version"])
            m = re.search(r"version:\s*(\S+)\s*\(build\s*(\d+),\s*commit\s*([0-9a-f]+)\)", out)
            if m:
                return {"tool": "llama.cpp", "version": m.group(1),
                        "build": m.group(2), "commit": m.group(3)}
            m = re.search(r"build\s*(\d+).*?commit\s*([0-9a-f]+)", out, re.S)
            if m:
                return {"tool": "llama.cpp", "build": m.group(1), "commit": m.group(2)}
    return {"tool": "llama.cpp", "build": "unknown", "commit": "unknown"}


def git_commit(path: Path) -> str:
    out = capture(["git", "-C", path, "rev-parse", "HEAD"]).strip()
    return out if re.fullmatch(r"[0-9a-f]{7,40}", out or "") else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="work")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base-dir", help="local HF model dir of the base model")
    g.add_argument("--base-repo", help="HF repo id to download (needs `hf` or `huggingface-cli`)")
    ap.add_argument("--adapter", default=str(Path(__file__).resolve().parent.parent
                                             / "adapter" / "qwen3.5-2b-heretic-r3"))
    ap.add_argument("--llama-cpp", required=True, help="dir containing llama-quantize[.exe]")
    ap.add_argument("--converter", default=None,
                    help="path to convert_hf_to_gguf.py (default: <llama-cpp>/convert_hf_to_gguf.py)")
    ap.add_argument("--gguf-py", default=None,
                    help="dir of gguf-py (default: sibling of the converter)")
    ap.add_argument("--merge-python", default=sys.executable,
                    help="python with torch+safetensors (for the merge step)")
    ap.add_argument("--pipeline-python", default=sys.executable,
                    help="python for convert/renorm/cmp/xpatch")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--codec", choices=["xz", "xz-cli", "zstd"], default="xz",
                    help="patch codec (default xz = stdlib lzma)")
    ap.add_argument("--gate", action="store_true",
                    help="quantize the base twice and compare (determinism gate)")
    ap.add_argument("--range", dest="range_patch", action="store_true",
                    help="also produce the range-format patch (attribution-capable)")
    ap.add_argument("--expect-base-sha", default=None, help="optional anchor to compare against")
    ap.add_argument("--expect-target-sha", default=None, help="optional anchor to compare against")
    args = ap.parse_args()

    work = Path(args.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    exe = ".exe" if os.name == "nt" else ""
    quantize = Path(args.llama_cpp) / f"llama-quantize{exe}"
    if not quantize.exists():
        sys.exit(f"FATAL: {quantize} not found")
    converter = Path(args.converter) if args.converter else Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    if not converter.exists():
        sys.exit(f"FATAL: converter not found: {converter}")
    gguf_py = Path(args.gguf_py) if args.gguf_py else converter.parent / "gguf-py"
    ppy, mpy = args.pipeline_python, args.merge_python

    # 0. base
    if args.base_repo:
        base = work / "base-model"
        if not base.exists():
            hf = shutil.which("hf") or shutil.which("huggingface-cli")
            if not hf:
                sys.exit("FATAL: neither `hf` nor `huggingface-cli` on PATH for --base-repo")
            run([hf, "download", args.base_repo, "--local-dir", base], label="download base")
    else:
        base = Path(args.base_dir).resolve()
        if not base.exists():
            sys.exit(f"FATAL: base dir not found: {base}")

    merged = work / "merged"
    base_f16 = work / "base-f16.gguf"
    var_f16 = work / "variant-f16.gguf"
    var_renorm = work / "variant-f16-renorm.gguf"
    base_q = work / f"base-{args.quant}.gguf"
    var_q = work / f"variant-{args.quant}.gguf"
    patch = work / "model-patch.patch.xz"
    manifest = work / "model-patch.manifest.json"
    provenance = work / "provenance.json"
    verify_out = work / "verify-tmp.gguf"

    # 1. surgical merge
    run([mpy, TOOLS / "merge_adapter_clean.py", "--base", base, "--adapter", args.adapter,
         "--out", merged], label="surgical merge")

    # 2. convert
    run([ppy, converter, base, "--outfile", base_f16, "--outtype", "f16"], label="convert base")
    run([ppy, converter, merged, "--outfile", var_f16, "--outtype", "f16"], label="convert merged")

    # 3. canonical metadata + structural check
    env = dict(os.environ)
    env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
    run([ppy, TOOLS / "gguf_renorm.py", var_f16, base_f16, var_renorm], env=env,
        label="metadata renorm")
    out = capture([ppy, TOOLS / "gguf_cmp.py", base_f16, var_renorm], env=env)
    ok = ("delta: 0" in out
          and "tensor type/shape diffs: 0" in out
          and "differing absolute data_offset: 0" in out)
    print(f"[check] structural alignment: {'OK' if ok else 'WARN — inspect manually'}")
    if not ok:
        print(out[-2000:])

    # 4. quantize both (+ optional determinism gate)
    run([quantize, base_f16, base_q, args.quant], label="quantize base")
    run([quantize, var_renorm, var_q, args.quant], label="quantize variant")
    if args.gate:
        base_q2 = work / f"base-{args.quant}-gate2.gguf"
        run([quantize, base_f16, base_q2, args.quant], label="quantize base (gate)")
        same = sha256_file(base_q) == sha256_file(base_q2)
        base_q2.unlink()
        print(f"[gate] determinism: {'PASS' if same else 'FAIL'}")
        if not same:
            sys.exit("FATAL: determinism gate failed — quantizer is not bit-reproducible here")
    base_sha, var_sha = sha256_file(base_q), sha256_file(var_q)
    print(f"[sha] base quant:    {base_sha}")
    print(f"[sha] variant quant: {var_sha}")

    # 4b. provenance
    base_st = next(iter(sorted(base.glob("*.safetensors"))), None)
    adapter_st = Path(args.adapter) / "adapter_model.safetensors"
    prov = {
        "base_model": {
            "name": args.base_repo or base.name,
            "source": args.base_repo or str(base),
            "safetensors": base_st.name if base_st else None,
            "sha256_safetensors": sha256_file(base_st) if base_st else None,
            "size_safetensors": os.path.getsize(base_st) if base_st else None,
        },
        "adapter": {
            "path": str(args.adapter),
            "sha256": sha256_file(adapter_st) if adapter_st.exists() else None,
        },
        "quantizer": {**llama_version(Path(args.llama_cpp)), "quant": args.quant},
        "converter": {"path": str(converter), "commit": git_commit(converter.parent)},
        "merge_python": {"torch": capture([mpy, "-c", "import torch; print(torch.__version__)"]).strip()},
        "pipeline_python": {"numpy": capture([ppy, "-c", "import numpy; print(numpy.__version__)"]).strip()},
    }
    provenance.write_text(json.dumps(prov, indent=2))
    print("[provenance]", json.dumps(prov["quantizer"]))

    # 5. XOR transport patch + provenance
    run([ppy, TOOLS / "xpatch.py", "make", "--base", base_q, "--target", var_q,
         "--patch", patch, "--manifest", manifest, "--provenance", provenance,
         "--codec", args.codec], label="xpatch make")

    # 6. verify: apply + cross-decoder apply
    run([ppy, TOOLS / "xpatch.py", "apply", "--base", base_q, "--patch", patch,
         "--manifest", manifest, "--out", verify_out], label="xpatch apply (verify)")
    verify_out.unlink()
    if args.codec in ("xz", "xz-cli") and shutil.which("xz"):
        other = "xz-cli" if args.codec == "xz" else "xz"
        cross_out = work / "verify-cross.gguf"
        run([ppy, TOOLS / "xpatch.py", "apply", "--base", base_q, "--patch", patch,
             "--manifest", manifest, "--out", cross_out, "--codec", other],
            label=f"xpatch apply (cross-check {other})")
        cross_out.unlink()
        if args.codec == "xz-cli" or other == "xz-cli":
            r = subprocess.run(["xz", "-t", str(patch)], capture_output=True)
            print(f"[check] xz -t integrity: {'OK' if r.returncode == 0 else 'FAIL'}")

    if args.range_patch:
        run([ppy, TOOLS / "diff_apply.py", "diff", base_q, var_q, "--gap", "16",
             "--patch", work / "model-patch.range.bin",
             "--manifest", work / "model-patch.range.json"], label="range diff")

    m = json.loads(Path(manifest).read_text())
    print("\n================ SUMMARY ================")
    print(f"base quant    : {base_q}  ({m['base_size']} B)")
    print(f"variant quant : {var_q}  ({m['target_size']} B)")
    print(f"patch         : {patch}  ({m['patch_bytes']} B = {m['patch_pct_of_base']}% of base)")
    print(f"codec         : {m['codec']}  ({m.get('compressor', '?')})")
    print(f"patch sha256  : {m['patch_sha256']}")
    print(f"variant sha256: {m['target_sha256']}")
    print(f"provenance    : {provenance}")
    anchors_ok = True
    if args.expect_base_sha and base_sha != args.expect_base_sha:
        print(f"[anchor] base sha MISMATCH\n  want {args.expect_base_sha}\n  have {base_sha}")
        anchors_ok = False
    if args.expect_target_sha and var_sha != args.expect_target_sha:
        print(f"[anchor] target sha MISMATCH\n  want {args.expect_target_sha}\n  have {var_sha}")
        anchors_ok = False
    print("ANCHORS OK" if anchors_ok else "ANCHORS: MISMATCH (see above)")
    print("DONE")


if __name__ == "__main__":
    main()
