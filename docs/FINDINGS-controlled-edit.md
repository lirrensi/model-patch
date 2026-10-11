# FINDINGS — GGUF diff/patch: end-to-end proof

**Date:** 2026-10-11 · **Model:** Qwen/Qwen3.5-2B (BF16) · **Toolchain:** llama.cpp build 11401 (commit a7fb71fab), Q4_K_M
**Status: PROVEN on this setup.** Every claim below is measured, not inferred. All artifacts in `work/`.

## Hypothesis

Given a model M quantized to Q (config pinned), and an edited variant M' quantized with the **same** config to Q':
the byte-difference patch(Q→Q') is small, applies losslessly (`Q + patch == Q'` byte-for-byte), and the result is a stock-legal GGUF.
Consequence: ship **one base + small patches** instead of N full model copies.

## Setup

- Source: Qwen3.5-2B safetensors (4,548,221,488 B) — hybrid arch: 24 layers (18 linear-attn + 6 full-attn), hidden 2048, MTP draft head (`blk.24`), VL wrapper (vision excluded → separate mmproj)
- Master: F16 GGUF, 3,897,387,968 B, 335 tensors, sha256 `08e8a296a6f7c8d9eb68f3f5d27e90418bcab47f1456cad5e60bee43f10554c6`
- Converter: `llama.cpp/convert_hf_to_gguf.py` (same llama.cpp tree, conversion/ package)
- Quantizer: `llama-quantize.exe` b11401, `Q4_K_M`, default flags (~25 s per run)
- Python: `.venv` (CPython 3.12.10, torch 2.12.0+cu130, numpy 2.5.3)

## The edit

- 7 tensors: `blk.{3,7,11,15,19,23,24}.attn_output.weight` (the o_proj family of this hybrid arch + MTP head)
- Rank-1 random unit direction, seed 42, per-tensor α chosen so ‖ΔW‖/‖W‖ = **2.000 %** exactly
- Applied in place on a copy of the F16 master (byte swap at tensor offsets; anchor-verified per tensor)

## Results

### Gate — quantization determinism

| file | sha256 | size |
|---|---|---|
| `A-q4km.gguf` | `c3276282e690d93f8f2f967fba2d7912f9986fbd21d2ad89e4eadd1b1c96dc2b` | 1,312,164,800 |
| `A2-q4km.gguf` (same input, same flags) | identical | identical |

→ llama-quantize b11401 is bit-reproducible (default threads). **Gate passed.**

### B (edited variant)

`B-q4km.gguf` sha256 `e42fc4ed7b40634483dfb4136fb3bff8d5fdadaf4f8ba478ff33f63b428a3143` — 1,312,164,800 B (same length as A).

### Diff A → B

- changed bytes: **4,288,690** = 0.327 % of file = **25.97 %** of the touched payload (16,515,072 B = 7 × 2,359,296)
- changed bytes **100 % inside the 7 touched tensors; 0 bytes outside** (no metadata/offset drift)
- per tensor: 25.5–27.0 %
- encodings measured: 2,193,254 raw runs → gap-16 coalesce → 143,182 ranges
  - gap-0 patch: 30,607,814 B — header overhead dominates, **rejected**
  - **gap-16 patch: `patch.bin` = 12,789,345 B = 0.975 % of the model** (payload 11,071,085 B + 1.72 MB headers)

### Apply proof

- verify base sha → copy → pwrite ranges → result sha256 **== B → PASS (byte-for-byte)** — run twice (gap-0 and gap-16 patches), both PASS
- reconstructed `patched.gguf` loads in stock tools: `llama-cli` (170 t/s prompt, 189 t/s gen), `llama-mtmd-cli` with mmproj

### Raw-differ comparison (zstd)

- `zstd --patch-from=A -19 --long=27` → 16,142,561 B (1.23 % of file); decode → sha == B ✓
- computed patch = **79 % of zstd's size**; zstd = 3.76× the raw changed bytes

### F16 master control

- master → master-edited: 33,671,423 changed bytes (0.864 % of 3.897 GB), patch 58,543,176 B (gap-16)
- (f16 payload per tensor is 3.6× the Q4_K payload — the quantized diff is the win)

### mmproj

- separate file from the same safetensors: `mmproj-Qwen3.5-2B-f16.gguf`, 668,227,136 B, sha256 `044a0ea136cca70711ae16e23b24d754b44eab6f2462d187aee4d7c7a9503d36`, 298 vision tensors
- not touched by text-side edits → **one mmproj serves base and all variants** (verified: loads with both A and patched)

## Proven claims

1. Quantization (b11401, Q4_K_M, default flags) is **bit-deterministic** — same input + flags → identical bytes.
2. Two quants of the same master family differ **only inside the edited tensors' payload ranges** — identical length, zero metadata/offset drift.
3. A computed `(offset, len, bytes)` patch applies **losslessly**: base + patch == target byte-for-byte (sha256-exact), result is a **stock-legal GGUF** (no engine changes, no forks).
4. Patch economics (this case): **12.79 MB patch = 0.975 %** of a 1.22 GiB model; raw changed 4.29 MB = 0.327 %.
5. Computed patch **beats zstd --patch-from** on the same pair (79 %).
6. mmproj is independent — text-side edits leave it bit-identical.

## Limits / requirements

- Patch validity requires a **bit-identical base**: same master + same quantizer build + same flags + same metadata (imatrix path strings etc.). Re-run the determinism gate after any build/backend change.
- Patch is sha-bound by design (base + target sha256 in the header); apply refuses on mismatch.
- Patches are **alternatives, not composable** (they overwrite the same byte ranges).
- Producer-side: quantize once to build the patch. Consumer-side: **base + patch + patcher only** — no quantizer, no second full-size download.
- **Cannot patch third-party quants** (their bytes are unreproducible) — works only inside your own artifact family.
- Transport is 100 % lossless; **semantic fidelity is a separate axis** (the lattice tax lives in the quantized edited weights, upstream of the diff).

## Not proven / open

- Semantic residual (thread: ~8.6 % at 1.57 % edit, Q4_0) not re-measured on this build/quant; random vector → no behavioral claim.
- Determinism across different machines/backends untested.
- Scale (9B / 26B) not yet run — expect ~linear in touched payload.
- imatrix metadata-drift scenario not exercised (no imatrix used).

## Artifacts (work/)

| file | size | sha256 |
|---|---|---|
| `master-f16.gguf` | 3,897,387,968 | `08e8a296…` |
| `master-edited-f16.gguf` | 3,897,387,968 | `271397b9dd7e38a9cea277eb36d0e7c233222b814825fa99c36698980dd487e6` |
| `A-q4km.gguf` | 1,312,164,800 | `c3276282…96dc2b` |
| `A2-q4km.gguf` | 1,312,164,800 | same as A |
| `B-q4km.gguf` | 1,312,164,800 | `e42fc4ed…28a3143` |
| `patched.gguf` | 1,312,164,800 | same as B |
| `patch.bin` | 12,789,345 | — (base/target sha embedded) |
| `master-edit.patch.bin` | 58,543,176 | — |
| `zstd-patch.bin` | 16,142,561 | — |
| `zstd-restored.gguf` | 1,312,164,800 | same as B |
| `mmproj-Qwen3.5-2B-f16.gguf` | 668,227,136 | `044a0ea1…503d36` |
| `manifest.json`, `master-edit-manifest.json` | — | diff stats |
| `edit-info.json` | — | per-tensor edit params (rel norm, α, offsets) |
| `master-tensors.json`, `A-tensors.json` | — | tensor maps |
| `edit_master.py`, `diff_apply.py`, `attribute.py` | — | tooling |

## Reproduce

```bash
cd <experiment-workspace>
# 0) venv (torch from local cache wheel)
uv venv .venv --python python
uv pip install --python python "<wheel-cache>/torch-2.12.0+cu130-cp312-cp312-win_amd64.whl" transformers safetensors numpy sentencepiece tqdm

# 1) convert + quantize base twice (determinism gate)
python llama.cpp/convert_hf_to_gguf.py Qwen3.5-2B --outfile work/master-f16.gguf --outtype f16
<llama.cpp>/llama-quantize.exe work/master-f16.gguf work/A-q4km.gguf Q4_K_M
<llama.cpp>/llama-quantize.exe work/master-f16.gguf work/A2-q4km.gguf Q4_K_M   # hashes must match

# 2) edit master + quantize edited
python tools/edit_master.py work/master-f16.gguf work/master-edited-f16.gguf --info work/edit-info.json
<llama.cpp>/llama-quantize.exe work/master-edited-f16.gguf work/B-q4km.gguf Q4_K_M

# 3) diff + apply + verify
python tools/diff_apply.py diff work/A-q4km.gguf work/B-q4km.gguf --gap 16 --patch work/patch.bin --manifest work/manifest.json
python tools/diff_apply.py apply --patch work/patch.bin --base work/A-q4km.gguf --out work/patched.gguf
# expected: "target sha256 MATCH" + "PASS"
```
