# model-patch

**Stop shipping the same weights twice.**

> TL;DR: original weights | quant + patch file => replaces having to store and download a 2nd pack of weights

Every abliteration, every decensoring, every small fine-tune — and the world downloads
another full multi-GB copy of a model it already has. The actual change is a few percent of
the bytes. The rest of the time those extra weights just sit on disk doing nothing: for each
model you hold the original *and* the variant. Twice the size, forever.

You use them once in a while, yet for every release you have to download 2 almost identical sets of weights!
If you use it only in 2% of workflows, that means it takes space both on your drive and on HF!

When you could just keep 5–10 % (worst case) of the original and materialize the 2nd set on demand.

This repo is the proof and the toolchain to do it properly: ship **base + small patch**, and
materialize the variant **on demand, byte-exact, sha256-verified** — then discard it when
you're done.

Worked example / test case — `Qwen3.5-2B` abliterated with a rank-3 LoRA (heretic defaults), quantized to Q4_K_M.
All numbers below are **measured** (receipts in `docs/`):

| what you ship | size | consumer needs |
|---|---|---|
| full variant GGUF (the current practice) | 1.22 GiB | nothing — and that's the waste |
| **base GGUF + XOR patch** | 1.22 GiB + **49.7 MiB (3.97 %)** | Python stdlib |
| base HF model + LoRA adapter | 4.2 GiB + **3.55 MB (0.08 %)** | torch + llama.cpp toolchain |

---

## The problem

Everyone who wants a variant downloads a second full set of weights. The difference between
the original and the variant is small — a few modified vectors per layer — yet entire
multi-gigabyte files are re-downloaded, re-hosted, re-stored. And you can't just "send the
difference" naively: the obvious tools (zstd/xdelta-style diffing, range-based patchers)
fall apart on quantized weights, because a quantized model's changed bytes are scattered as
*millions of tiny fragments*. 

Measured on the worked example: 60.1 MB of changed bytes in
**27,470,832 fragments** (avg 2.19 bytes each). Fragment-addressing formats pay 12 bytes of
bookkeeping per fragment and explode:

| method | patch size | % of model |
|---|---|---|
| range diff, best gap (16 B) | 149.3 MiB | 11.93 % |
| range diff, gap 0 + zstd | 118.3 MiB | 9.0 % |
| off-the-shelf `zstd --patch-from` | 162.5 MiB | 12.99 % |
| **XOR transport + xz (this repo)** | **49.7 MiB** | **3.97 %** |

XOR transport wins in every regime we measured — dense quantized diffs (3.0× smaller than
the best range variant), sparse Q4 edits (3.16×), sparse F16 edits (1.71×). It even lands
*below* the raw changed-byte count: identical bytes become `0x00` (free for the entropy
coder), and the changed-byte XORs themselves compress ~13 %.

## Stage 1 — brute-force binary diff (no ML dependencies)

The trust-free baseline. No LoRA loading, no framework behavior to worry about — pure bytes:

1. Get the original safetensors / GGUF (the base).
2. Produce the modified set of weights (once, by whoever makes the variant).
3. Diff base against the modified set; keep a single binary patch file.
4. Discard the modified set. Anyone can now materialize it from base + patch at any moment —
   load it, use it, discard it again.

The patch format (schema `model-patch/1`) is deliberately dumb, so it can't break:

```
patch      = <codec>( base XOR variant )            # codec: xz (stdlib) | xz-cli | zstd
manifest   = {
  schema: "model-patch/1",
  base_sha256, target_sha256, base_size, target_size,
  codec, compressor,                     # e.g. "python-lzma preset=9|EXTREME (python 3.12.10)"
  patch_sha256, patch_bytes, patch_pct_of_base, xor_stream_bytes,
  created,
  provenance: {                          # supplied by the producer — document everything
    base_model:  { name, source, revision, sha256_safetensors, size_safetensors },
    adapter:     { path, sha256 },
    quantizer:   { tool: "llama.cpp", build: "11401", commit: "a7fb71fab", quant: "Q4_K_M" },
    converter:   { path, commit },
    merge_python: { torch }, pipeline_python: { numpy }
  }
}
apply      = verify base → verify patch → decompress → XOR → truncate → verify target
```

Consumer side:

```bash
python tools/xpatch.py apply --base base.gguf --patch model-patch.xz \
    --manifest model-patch.manifest.json --out variant.gguf
# → base sha256 OK / patch sha256 OK / target sha256 MATCH / PASS
```

**Requirements: Python stdlib only.** (xz via the stdlib `lzma` module — no external
binaries, no numpy needed, no ML libraries, no GGUF knowledge. The apply path never parses
the file; it is format-blind byte work.) `--codec xz-cli` uses the standalone `xz` binary
instead — an **independent decoder** that cross-checks the stdlib path (both decode each
other's files byte-identically; verified). Works for equal and differing sizes.

## Stage 2 — exact vectors (LoRA adapter + script)

The smaller route: instead of a full binary difference, ship the **exact change vectors**
(a LoRA adapter) plus **a script that applies them**. Smaller than the binary patch
(3.55 MB vs 49.7 MiB here), but it requires torch on the consumer side and per-model
adaptation — less universal. For the consumer it must be an exact process:
*run a script → get the 2nd set of weights.*

### Why you must apply the LoRA yourself (measured — never trust the trainer's export)

**Rule: use heretic (or whatever tool you use) only to produce the adapter — never trust it
to save the merged model for you.** The adapter format is tool-agnostic (a PEFT LoRA is a
PEFT LoRA): take the adapter and apply it carefully yourself, as this repo does. Here is the
measured failure from when we let the tool do the merge — its export must never be used as a
diff base:

- **drops the MTP head** — 15 tensors, 121.66 MB, gone;
- **casts F32 tensors to BF16** (36 tensors) — silent precision change;
- **metadata drift** — `general.name`/`general.finetune` derive from the directory name and
  README, so the exported GGUF's metadata differs → `data_offset` shifts by −1312 bytes →
  **every one of the 320 common tensors is displaced** → any byte-level diff becomes the
  whole file.

So this repo ships its own merge script (`tools/merge_adapter_clean.py`) and applies the
adapter surgically:

- output = **byte-copy** of the original safetensors;
- only the LoRA's target tensors are rewritten: `W' = (W.f32 + (B.f32 @ A.f32)).to(orig_dtype)`;
- everything else — MTP, F32 tensors, the safetensors header — stays byte-identical;
- result: 37 of 48 targets changed; the diff is clean and small.

And because GGUF metadata is derived from the model directory name / README, the pipeline
**canonicalizes metadata** (`tools/gguf_renorm.py`): the variant's GGUF gets the base's exact
metadata KV section, so tensor offsets align (verified with `tools/gguf_cmp.py`: 0 field
diffs, 0 shape diffs, 0 offset diffs).

## Quantization — same config or bust (measured)

Quantization is diffed the same way — but it has a hard limitation:

- **Quantize both versions yourself with the same config, and keep the config stable between
  the two runs.** Any drift (quant type, flags, metadata, a different llama.cpp commit)
  changes the bytes rapidly — measured: a −1312-byte metadata shift alone displaced every
  tensor and turned a 149 MiB patch into a whole-file patch.
- **You cannot take an arbitrary quant, apply a patch, and expect it to work.** The base
  quant and the patch must come from the same source/pipeline — the person who published the
  base quant is the one who must build the patch. The good news: it is **bit-reproducible** —
  quantizing the same input twice with the same build produced identical sha256, across
  sessions (gate re-proven 7 hours apart).
- Quant strategies beyond the default (custom layer-wise configs) can be shared as a config +
  a pinned llama.cpp build, and everyone will reproduce the same quant bytes.

## The pipeline (order of steps, as implemented)

0. **Get the base** — HF model dir (`Qwen/Qwen3.5-2B`) or a local copy.
1. **Surgical merge** — `tools/merge_adapter_clean.py` applies the shipped adapter
   (or any PEFT LoRA) to the base safetensors. *(needs: python + torch + safetensors)*
2. **Convert both to F16 GGUF** — `convert_hf_to_gguf.py` from the llama.cpp tree, same
   converter for base and variant. *(needs: python + transformers + the converter tree)*
3. **Canonicalize variant metadata** — `tools/gguf_renorm.py` copies the base's metadata KV
   verbatim; `tools/gguf_cmp.py` proves alignment (expect: delta 0, 0 diffs). *(needs: gguf-py)*
4. **Quantize both** — `llama-quantize` with the same build and flags (worked example:
   Q4_K_M, build b11401 / a7fb71fab — pinned; a different build will produce different bytes).
5. **Make the patch** — `tools/xpatch.py make` (XOR transport) with a **provenance manifest**.
6. **Verify** — `tools/xpatch.py apply` to a scratch file, sha256 vs the manifest target,
   **plus a cross-decoder apply** (xz CLI vs stdlib lzma). **PASS or it didn't happen.**

`pipeline/reproduce.py` runs 0–6 end-to-end with checks:

```bash
python pipeline/reproduce.py \
    --workdir work \
    --base-repo Qwen/Qwen3.5-2B \
    --adapter adapter/qwen3.5-2b-heretic-r3 \
    --llama-cpp /path/to/llama.cpp/bin \
    --converter /path/to/llama.cpp/convert_hf_to_gguf.py \
    --merge-python /path/to/venv-with-torch/bin/python \
    --pipeline-python /path/to/venv-with-numpy/bin/python \
    --quant Q4_K_M --gate
```

`--gate` additionally quantizes the base twice and compares (the determinism gate).
`--range` also emits the range-format patch (useful for byte-range → tensor attribution via
`tools/attribute.py`). `--codec xz-cli` uses the external xz binary for the patch.
The run writes `work/provenance.json` and embeds it into the patch manifest.

## What you need

| role | needs |
|---|---|
| **consumer, patch route** | Python 3.8+ (stdlib only) + base model + patch + manifest |
| consumer, adapter route | torch + safetensors (merge), transformers + llama.cpp converter, llama-quantize (pinned build), F16 base — minutes of compute |
| **producer** | all of the above, plus: the variant's LoRA adapter (or any modified weights), the same quantizer build/flags the base quant was made with |
| optional | numpy (speed in xpatch), `xz`/`zstd` binaries (alternate codecs) |

## Tool differences that can bite you (all measured — document them, ship them)

1. **Compressor tool/version changes the patch bytes.** Same input, same content:
   `xz -9e -T8` (CLI) → 52,145,312 B; stdlib `lzma` preset 9e → 52,135,304 B. Different
   compressed bytes, identical decompressed content — decoders are interchangeable
   (verified cross-decoder: py-lzma and xz CLI produced the *same stream sha* on the same
   patch, and `xz -t` validates both). The manifest pins `patch_sha256`, so consumers can
   always tell which bytes they received. Never compare patches by size — compare by sha.
2. **llama.cpp build/commit changes the quant bytes.** The patch is bound to the exact base
   quant (sha-gated, fail-closed). Ship the quantizer build + commit in `provenance` — the
   worked example records `build 11401 / commit a7fb71fab` and it is bit-reproducible across
   sessions on the same build.
3. **Converter and merge-runtime versions** can shift GGUF bytes (converter: tensor order,
   metadata; merge: torch matmul behavior). `provenance` records converter commit + torch +
   numpy versions; the merge runs fp32 math with a single thread for stability.
4. **The environment itself can flake.** During this project we observed two isolated
   single-byte corruptions on a machine with a history of BSODs (one in a make's XOR stream,
   one in an apply's output — caught immediately by the sha gates, different offsets, not
   reproducible on rerun; `xz -t` and cross-decoder checks were clean). Lesson: **every
   step is sha-gated on both ends, and the verify step is mandatory** — silent corruption cannot
   ship, because base, patch, and target hashes all have to agree. If a verify fails on a
   flaky host: rerun; the pipeline is idempotent.
5. **Do not trust "it's the same model" without hashes.** Dir names lie (`general.name`
   derives from them), README frontmatter leaks into GGUF metadata, and merge tools silently
   drop tensors. The pipeline's answer: hash everything, renorm metadata, and verify.

## Verification anchors (worked example)

Reproduce these and you've reproduced our bytes exactly. Different llama.cpp builds or
converter versions may legitimately produce different bytes — in that case use *your own*
pipeline's shas consistently (base quant + patch from the same source, always).

| artifact | size (B) | sha256 |
|---|---|---|
| base `model.safetensors` (BF16) | 4,548,214,832 | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |
| adapter (r=3, α=3, 96 tensors) | 3,553,000 | `bcd2efafa6a03529b3dbdab09a776cc67be200872ff720f65e9b190e88cc015d` |
| surgically merged `model.safetensors` | 4,548,214,832 | `532312bfa7dd96146ff7bd281bb867640747bcd4ebb119c0413fb6121889162b` |
| base F16 GGUF (335 tensors) | 3,897,387,968 | `08e8a296a6f7c8d9eb68f3f5d27e90418bcab47f1456cad5e60bee43f10554c6` |
| base Q4_K_M GGUF | 1,312,164,800 | `c3276282e690d93f8f2f967fba2d7912f9986fbd21d2ad89e4eadd1b1c96dc2b` |
| variant Q4_K_M GGUF | 1,312,164,800 | `88018427df3ed435a258d13d17970fc75d2dbcf307dba9d711bf0e64e063f59e` |
| **patch (xz, stdlib lzma)** | **52,135,304** | `0fa1298e598d5d136763067a5955651039d559d8b3336a08aa0c48419708015b` |
| patch (xz CLI -T8 variant) | 52,145,312 | `b2ffad47726cb9863d4822d11985dd6e33ffb624b7030a17a5931f0e3a6359ad` |

The end-to-end proof (for the record): the variant quant was **deleted**, then reconstructed
from base + patch — `target sha256 MATCH` — and the reconstructed model loads and generates
in stock `llama-cli`. Full records: `docs/FINDINGS-abliterated-diff.md` (real abliteration)
and `docs/FINDINGS-controlled-edit.md` (controlled small edit + the determinism gate).

## Layout

```
adapter/qwen3.5-2b-heretic-r3/   the exact LoRA used + model-patch.manifest.json and
                                 provenance.json for the published patch (no heretic needed)
pipeline/reproduce.py            full-path reproduction: base + adapter → patch (steps 0–6)
tools/xpatch.py                  XOR transport: make / apply (consumer: stdlib only)
tools/diff_apply.py              range-format patcher v2 (attribution, sparse edits)
tools/attribute.py               patch byte-ranges → GGUF tensors
tools/merge_adapter_clean.py     surgical LoRA merge (byte-clean)
tools/gguf_renorm.py             canonical metadata rewrite
tools/gguf_cmp.py                structural comparator (offsets/fields/tensors)
docs/                            full experiment records with all measurements
```

The patch itself (`model-patch.patch.xz`, 52,135,304 B, sha `0fa1298e…`) ships as a
**release asset** — kept out of the repo (regenerable byte-exactly via
`pipeline/reproduce.py`, verifiable via the manifest).
→ **[Download v1.0.0](https://github.com/lirrensi/model-patch/releases/tag/v1.0.0)**

## Limits / honest notes

Major limitation: it's easy to provide a diff to an original model, but for quants you have to both quantize and make a patch yourself.
You can't take an Unsloth quant and patch it (maybe with a different tool — this concerns .safetensors first).

- **Same-source rule:** a patch only applies to the exact base it was built from (sha-bound,
  fail-closed). That's a feature: no silent mismatches.
- Determinism scope: same machine class + same llama.cpp build/flags + same converter. The
  pipeline pins what it can and verifies what it can't.
- The 3.97 % figure is for a real abliteration (LoRA rank 3) on Qwen3.5-2B at Q4_K_M. Other
  edits/models will differ; the *method* is what generalizes, and `reproduce.py --range`
  + `attribute.py` will tell you exactly which tensors changed and how much.
- **Never trust the abliteration tool to save the full model.** Produce the adapter with
  whatever tool; apply the merge yourself (`tools/merge_adapter_clean.py`) — measured failure
  case above: dropped tensors, precision casts, metadata drift, every offset shifted.

Best case: ship both the LoRA adapter + the script to apply it, a diff for specific quants, AND your quantization config.
That allows the best of both worlds — anyone who needs raw safetensors for vLLM has everything already.
And if you provide the quant config when it's not the default, everyone can reproduce the full chain.
And most users download Q3/Q4/Q6/Q8 quants — you can make small diffs for each.

For llama.cpp, mmproj parts typically stay the same, so you don't even need to include them in the patch files.

## Please

Save space on both your drive and on HF! They bleed money because we can't use very simple tools and waste space and bandwidth pointlessly!

If users can understand how to run a model in the CLI, they can run a single script to make a copy of the weights!

## License
MIT for all code; `Qwen3.5-2B` — Apache 2.0.