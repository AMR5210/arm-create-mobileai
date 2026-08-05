# Running Phases 0–3 on Google Colab (free GPU)

Colab's free tier is a reasonable substitute for the plan's "GPU cloud
instance" role (Phase 3 QAT training), and can also run everything upstream
of it (Phase 0 setup, Phase 1 fp16 conversion, Phase 2 PTQ), ending with three
downloadable `.gguf` files: `qwen3-0.6b-fp16.gguf`, `qwen3-0.6b-ptq-q2_k.gguf`,
`qwen3-0.6b-qat-q2_k.gguf`.

**What Colab must NOT be used for:** reported benchmark numbers. Colab gives
you an x86_64 Linux VM with a GPU (typically a T4 on the free tier) — nothing
like an iPhone's Arm CPU. Running `scripts/benchmark.py` there would produce
numbers that are meaningless for the submission (Section 5 of
`docs/METHODOLOGY.md`: final numbers must come from the iPhone 17 Pro Max).
Colab's job here is purely to produce the three `.gguf` files; benchmarking
those files still happens later on the Mac mini (dev/sanity-check) and
iPhone (reported numbers).

## Gaps this required fixing in the scripts (already done)

1. **`scripts/setup_env.sh` — venv doesn't persist across Colab cells.**
   Each `!`-prefixed Colab cell runs in its own fresh subshell, so a venv
   activated inside a script wouldn't carry over to the next cell — later
   `!python ...` calls would silently use the system Python and miss
   everything installed in the venv. The script now detects Colab
   (`COLAB_RELEASE_TAG` env var, or the presence of `/content`) and installs
   directly into the system Python in that case, skipping the venv.

2. **`train_qat.py` only saved once, at the very end.** Free Colab sessions
   disconnect often enough (idle timeouts, session limits) that a multi-hour
   training run with a single save-at-the-end is a real risk of losing
   everything. Added `--save-every` (default every 100 steps), which writes a
   *non-destructive* safety-net checkpoint (current fake-quantized weights,
   via the new `qat.apply_qat.fake_quantized_state_dict`) to
   `<output-dir>-partial` without disturbing the live training loop. If the
   session dies mid-run, that directory has a directly-usable (if partially
   trained) HF checkpoint — no optimizer state, so it's not a true resume,
   but it's a usable fallback instead of a total loss.

## What already works without changes

- **`scripts/build_llama_cpp.sh`** already branches on `uname -m`: on Colab's
  x86_64 host it takes the "non-Arm" path and skips `-DGGML_CPU_KLEIDIAI=ON`
  automatically — expected and correct, not an error. (KleidiAI is
  Arm-only and irrelevant to producing the `.gguf` files anyway.)
- **`scripts/download_model.py`**, **`scripts/prepare_eval_data.py`**,
  **`scripts/convert_to_gguf.sh`**, **`scripts/quantize_ptq.sh`**,
  **`scripts/export_qat_gguf.sh`** — all plain Python/bash with no
  macOS/Arm-specific assumptions, all network-dependent steps use plain
  HTTPS (HF Hub, HF `datasets`) which Colab has no trouble with.
- **`Accelerator()`** in `train_qat.py` degrades gracefully to single-device
  on a single Colab GPU — just run `python scripts/train_qat.py ...` directly,
  no `accelerate launch` or config file needed.

## Suggested Colab notebook flow

```python
# 1. Mount Drive and clone into it, so downloaded/converted artifacts survive
#    a disconnect (Colab's local /content disk is ephemeral).
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone <your-repo-url> arm-create-mobileai
%cd arm-create-mobileai
!git checkout claude/project-implementation-p4rkuv
```

```bash
%%bash
# 2. Install dependencies. Colab already ships a CUDA-enabled torch --
# reinstalling it from requirements.txt would just waste time re-downloading
# a large wheel, so install the rest directly instead of the full file.
pip install -q transformers accelerate datasets huggingface_hub sentencepiece protobuf tqdm
apt-get -qq install -y cmake build-essential  # usually already present, harmless if so
```

```bash
%%bash
# 3. Build llama.cpp (CPU build, non-Arm host -- expected, see above)
scripts/build_llama_cpp.sh

# 4. Download the base model, prepare eval data
python scripts/download_model.py
python scripts/prepare_eval_data.py

# 5. fp16 GGUF (this is the "baseline-fp16" artifact)
scripts/convert_to_gguf.sh models/qwen3-0.6b-hf models/qwen3-0.6b-fp16.gguf

# 6. PTQ 2-bit GGUF (this is the "ptq-2bit" artifact)
scripts/quantize_ptq.sh models/qwen3-0.6b-fp16.gguf models/qwen3-0.6b-ptq-q2_k.gguf
```

```bash
%%bash
# 7. QAT fine-tune on the free GPU. --save-every protects against disconnects.
# Watch the wikitext2_ppl trend printed every --eval-every steps; stop early
# (interrupt the cell) once it plateaus rather than burning free-tier hours.
python scripts/train_qat.py \
  --base-model models/qwen3-0.6b-hf \
  --output-dir models/qwen3-0.6b-qat-hf \
  --max-steps 500 \
  --save-every 100 \
  --eval-every 50

# 8. Export QAT weights to the same GGUF Q2_K format as the PTQ baseline
# (this is the "qat-2bit" artifact)
scripts/export_qat_gguf.sh models/qwen3-0.6b-qat-hf
```

```python
# 9. Download the three .gguf files. Since step 1 cloned into Drive, they're
# already synced to your regular Google Drive too -- files.download() just
# gives you a direct browser download on top of that.
from google.colab import files
files.download('models/qwen3-0.6b-fp16.gguf')
files.download('models/qwen3-0.6b-ptq-q2_k.gguf')
files.download('models/qwen3-0.6b-qat-q2_k.gguf')
```

## Other things worth knowing before you hit run

- **GPU memory**: a 0.6B model trained in full fp32 (all parameters, not a
  small adapter) needs room for params + gradients + AdamW's two fp32 moment
  buffers, plus activations. Should fit a free-tier T4 (16GB) at the default
  `--batch-size 4 --max-length 512`, but if you hit a CUDA OOM, lower
  `--batch-size` first (no code changes needed, both are existing CLI flags).
- **Thermal/quota throttling isn't a concern here** (that's a mobile-hardware
  issue, see `ios/README.md`) — Colab's own
  limit is session/usage quotas on the free tier, which `--save-every`
  mitigates but doesn't eliminate. Don't leave a run going unattended for
  hours expecting it to definitely finish.
- **`--max-steps`, `--eval-every`, `--dataset`** etc. are all already
  CLI-adjustable in `scripts/train_qat.py` -- no code changes needed to tune
  the run itself, only to fix the two Colab-specific gaps above.
