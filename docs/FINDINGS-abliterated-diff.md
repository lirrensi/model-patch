# FINDINGS — real abliteration transported as base + patch (2026-10-11)

Extends the controlled-edit proof (`FINDINGS.md`) to a **real weight change**: heretic
abliteration (LoRA, r=3) on Qwen3.5-2B, quantized Q4_K_M with llama.cpp b11401 (a7fb71fab).

## Result

- Target **B** (abliterated Q4_K_M) was **deleted after hashing**; reconstructed from
  **A** (original Q4_K_M) + `abl-clean.patch.bin` (**156,590,703 B = 11.93 %** of the model)
  → **sha256 MATCH (PASS)**.
- **Transport v2 (XOR+xz, `xpatch.py`)**: same reconstruction via `xpatch-q4km.patch.xz` =
  **52,145,312 B = 3.97 %** (patch sha `b2ffad47…`) → **sha256 MATCH (PASS)**.
  3× smaller than the range format; below even the raw changed-byte count (57.3 MiB) because
  identical bytes become zero (free) and the delta bytes themselves compress ~13 %.
- Patch attribution (range format): **0 bytes outside the abliterated tensors**.
- Producer pipeline required: heretic's own merged export is NOT usable as-is (measured, below)
  → custom **surgical adapter merge** + **GGUF metadata renorm**.

## Hashes & sizes

| artifact | size (B) | sha256 |
|---|---|---|
| original `model.safetensors` (BF16, 632 tensors) | 4,548,214,832 | (source) |
| adapter `adapter_model.safetensors` (r=3, α=3) | 3,553,000 | `bcd2efafa6a03529b3dbdab09a776cc67be200872ff720f65e9b190e88cc015d` |
| clean merged `model.safetensors` | 4,548,214,832 | `532312bfa7dd96146ff7bd281bb867640747bcd4ebb119c0413fb6121889162b` |
| `master-f16.gguf` (base, 335 tensors) | 3,897,387,968 | `08e8a296a6f7c8d9eb68f3f5d27e90418bcab47f1456cad5e60bee43f10554c6` |
| **A** = `orig-q4km.gguf` | 1,312,164,800 | `c3276282e690d93f8f2f967fba2d7912f9986fbd21d2ad89e4eadd1b1c96dc2b` |
| **B** = `abl-clean-q4km.gguf` (**deleted after hashing**) | 1,312,164,800 | `88018427df3ed435a258d13d17970fc75d2dbcf307dba9d711bf0e64e063f59e` |
| `abl-clean.patch.bin` (range format) | 156,590,703 | `a2d0262e233ba2196dfd8629735446a74f455f38175c27318b8aa0bb39a056f0` |
| `abl-clean-reproduced.gguf` (range format) | 1,312,164,800 | `88018427…` (== B ✓) |
| `xpatch-q4km.patch.xz` (XOR transport) | **52,145,312** | `b2ffad47726cb9863d4822d11985dd6e33ffb624b7030a17a5931f0e3a6359ad` |
| `xpatch-reproduced.gguf` (XOR transport) | 1,312,164,800 | `88018427…` (== B ✓) |

## Measurements

- **Determinism gate (re-proof, cross-session):** fresh quantize of `master-f16.gguf`
  → sha `c3276282…96dc2b`, identical to the 7-hour-old `A-q4km.gguf`.
- **Diff** (gap 16): 1,507,451 ranges; raw changed **60,087,853 B (4.5793 %)**;
  payload **138,501,199 B (10.5552 %)**; patch total **156,590,703 B (11.9338 %)**; 16.69 s.
- **Attribution** (`\.(attn_output|ffn_down|ssm_out)\.weight$`):
  inside touched **138,501,199 B (49.80 % of 278,102,016 B touched payload)**;
  outside **0**; ranges fully outside: **0**.
  37/48 abliteration targets nonzero (65–75 % of each tensor's Q4 payload);
  11 optimizer-zero targets + 2 MTP tensors = 0 bytes.
- **Merge level (BF16):** 37 tensors changed, 257,794,179 B (42.10 % of their payload);
  safetensors header byte-identical; MTP + F32 tensors untouched.
- **Abliteration scope:** optimizer zeroed layers 0–5 (mostly) and ablated layers 5–23 —
  "update only a fraction of layers, else KL explodes" (expected heretic behavior).

## Transport format comparison (measured on this pair)

Raw structure: the 60.1 MB of changed bytes are **27,470,832 tiny fragments** (avg 2.19 B,
~45 B apart) — quantized-weight flips inside 6-bit quant slots.

Range format (`diff_apply.py`, 12 B/range records):

| gap | ranges | patch (B) | % of model | patch + zstd -19 |
|---|---|---|---|---|
| 0 | 27,470,832 | 389,737,929 | 29.70 % | 118.3 MiB |
| **16** | 1,507,451 | 156,590,703 | **11.93 %** | 138.2 MiB |
| 64 | 145,030 | 179,210,595 | 13.66 % | 167.7 MiB |
| 256 | 11,808 | 190,519,651 | 14.52 % | 178.7 MiB |
| 1024 | 1,738 | 194,968,493 | 14.86 % | 182.9 MiB |

- gap 16 was near-optimal for the 12-B-record format (records↔fill tradeoff).
- zstd helps only the g0 patch (the record stream compresses): 371.7 → 118.3 MiB —
  still 2.4× worse than XOR transport.

XOR transport (`xpatch.py`): patch = codec(A XOR B), streaming apply + sha256 verification:

| codec | size (B) | % of model |
|---|---|---|
| **xz, stdlib `lzma` 9e (single-thread)** | **52,135,304** | **3.9732 %** |
| xz -9e -T8 (CLI) | 52,145,312 | 3.974 % |
| zstd -19 -T8 --long=27 | 53,535,894 | 4.08 % |

- off-the-shelf `zstd --patch-from` baseline: 170,390,856 B (12.99 %) — **worse than our
  range format**; its sequence coding is inefficient for 27.5 M tiny fragments.
- raw changed bytes: 60,087,853 (4.58 %) — XOR+xz beats it (delta bytes ~13 % compressible,
  zero runs free).
- **F16 control** (gap 16): 599,701,962 B (15.39 %); raw changed 304,426,719 (7.81 %)
  → Q4 diffing is **3.8× cheaper** than F16 for the same edit.
- XOR verification: `make` → `apply` → `base sha OK`, `patch sha OK`, `target sha MATCH`
  `88018427…` → **PASS** (patch sha `b2ffad47…`).
- Stdlib-only verification: `XPATCH_NO_NUMPY=1 xpatch.py apply` (pure-Python XOR + stdlib
  `lzma`) on the xz-CLI-made patch → **PASS**; stdlib-made patch (`0fa1298e…`) → **PASS**;
  interop both directions with the xz CLI (`xz -t` OK).

Cross-check on the earlier controlled-edit pairs (XOR transport wins there too):

| pair | raw changed | range patch (gap 16) | XOR+xz | XOR advantage |
|---|---|---|---|---|
| Q4 sparse (exp-1: A-q4km vs B-q4km) | 4,288,690 | 12,789,345 | **4,048,012** | 3.16× |
| F16 sparse (exp-1: master vs master-edited) | 33,671,423 | 58,543,176 | **34,303,304** | 1.71× |

Conclusion: XOR+xz supersedes the range format for whole-file transport in **all** measured
regimes (dense and sparse); range patches remain useful for attribution.

## Determinism & environment notes (2026-10-11)

- `xpatch.py make` is **bit-deterministic**: two fresh makes from the same base/variant
  content produced the identical patch sha (`0fa1298e…`), matching an earlier run from
  different paths. `apply` is byte-exact (`output == base XOR stream`, 0 diffs).
- Decompression is deterministic and **cross-decoder identical**: stdlib `lzma` and the
  `xz` CLI (5.8.3) produce the same decompressed stream sha; `xz -t` validates both.
- During the pipeline's first full self-test, the host (a machine with a history of BSODs)
  produced two **transient single-byte corruptions** — one inside a make's XOR payload
  (offset 257,976,062), one inside an apply's output (offset 1,074,736,318). Both were
  caught immediately by the sha256 gates; both vanished on rerun (fresh makes: identical
  sha to the original good patch). Lesson: **verify is mandatory**; the format is
  self-checking (base + patch + target hashes); flaky hosts can rerun safely.

## Why heretic's own merged export fails as producer (measured on `Qwen3.5-2B-A-Full`)

1. **drops all 15 `mtp.*` tensors** (121,656,320 B) → tensor sets differ.
2. casts F32 tensors (`ssm_a`, `ssm_norm` ×18) to BF16 (converter normalizes back, but values
   roundtrip through BF16).
3. **metadata drift** (dir-name-derived `general.name`/`general.finetune`, README-derived
   fields) → `data_offset` delta **−1312** → **all 320 common tensors shifted** →
   naive diff ≈ whole file.

Conclusion: the producer must (a) merge the adapter **surgically** (only target tensors,
everything else byte-copied) and (b) **canonicalize GGUF metadata** (renorm) before quantizing.

## Reproduce

```bash
# 0) surgical merge (H venv has torch+safetensors)
python tools/merge_adapter_clean.py \
    --base Qwen3.5-2B --adapter adapter/qwen3.5-2b-heretic-r3 --out Qwen3.5-2B-A-Clean

# 1) convert clean model (F16)
python llama.cpp/convert_hf_to_gguf.py Qwen3.5-2B-A-Clean \
    --outfile work/abl-clean-f16.gguf --outtype f16

# 2) canonicalize metadata against the base
PYTHONPATH=llama.cpp/gguf-py python tools/gguf_renorm.py \
    work/abl-clean-f16.gguf work/master-f16.gguf work/abl-clean-f16-renorm.gguf
PYTHONPATH=llama.cpp/gguf-py python tools/gguf_cmp.py \
    work/master-f16.gguf work/abl-clean-f16-renorm.gguf  # expect: delta 0, 0 diffs

# 3) quantize both, same binary/flags
<llama.cpp>/llama-quantize.exe work/master-f16.gguf work/orig-q4km.gguf Q4_K_M
<llama.cpp>/llama-quantize.exe work/abl-clean-f16-renorm.gguf work/abl-clean-q4km.gguf Q4_K_M

# 4) range diff, record hash, DELETE target, apply, verify (attribution-capable)
python tools/diff_apply.py diff work/orig-q4km.gguf work/abl-clean-q4km.gguf \
    --gap 16 --patch work/abl-clean.patch.bin --manifest work/abl-clean-manifest.json
sha256sum work/abl-clean-q4km.gguf
rm work/abl-clean-q4km.gguf
python tools/diff_apply.py apply --patch work/abl-clean.patch.bin \
    --base work/orig-q4km.gguf --out work/abl-clean-reproduced.gguf   # → PASS

# 5) XOR transport (preferred, 3x smaller)
python tools/xpatch.py make --base work/orig-q4km.gguf \
    --target work/abl-clean-reproduced.gguf --patch work/xpatch-q4km.patch.xz \
    --manifest work/xpatch-q4km.json
python tools/xpatch.py apply --base work/orig-q4km.gguf \
    --patch work/xpatch-q4km.patch.xz --manifest work/xpatch-q4km.json \
    --out work/xpatch-reproduced.gguf   # → PASS

# 6) attribute (range patch)
python tools/attribute.py --patch work/abl-clean.patch.bin \
    --tensors work/A-tensors.json --pattern '\.(attn_output|ffn_down|ssm_out)\.weight$'
```

## Tools (state after this session)

| tool | role |
|---|---|
| `merge_adapter_clean.py` | surgical PEFT LoRA merge (fp32 accumulate; everything else byte-copied) |
| `gguf_renorm.py` | rewrite GGUF with reference metadata KV (canonical metadata) |
| `gguf_cmp.py` | structural comparator (data_offset, field sizes, tensor map; writes attribute maps) |
| `diff_apply.py` | v2 `GGPATCH2` range patcher (size-aware; reads v1); kept for sparse edits + attribution |
| `attribute.py` | v2-aware range→tensor attribution |
| `xpatch.py` | **XOR transport**: patch = xz/zstd(A XOR B); streaming apply; manifest with sha256s; **stdlib-only** (xz via Python `lzma`; numpy optional for speed) |
| `xor_stream.py`, `sweep_report.py` | experiment utilities (XOR dump, sweep summary) |
| `edit_master.py` | (previous experiment) in-place master edit |

## Notes / limits

- Range format is superseded for transport (measured in all three regimes: dense Q4 3.0×,
  sparse Q4 3.16×, sparse F16 1.71× smaller with XOR+xz). Use `xpatch.py` for shipping;
  range patches remain useful for attribution (byte-ranges → tensors).
- renorm is needed because converter metadata derives from dir name/README; a production
  producer should pin `--model-name` + identical sidecars, or always renorm.
- determinism scope: same machine, same llama.cpp build (b11401) and flags.
- renorm bug history (fixed): (a) reader pseudo-fields `GGUF.*` must never be copied into KV;
  (b) `raw_shape` must be passed in numpy order (writer reverses when writing `ne[]`) —
  transposed shapes caused 36 tensors to fall back to different quant types.
