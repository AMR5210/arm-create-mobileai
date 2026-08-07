# Running Phase 3 (QAT training) on an AMD ROCm cloud GPU

This is an alternative to Google Colab (`docs/COLAB_SETUP.md`) for the plan's
"GPU cloud instance" role. PyTorch ships official ROCm builds that expose the
same `torch.cuda` API ROCm uses internally for AMD GPUs, so **nothing in this
repo's code needs to change** — `Accelerator()` in `scripts/train_qat.py`
auto-detects the device the same way it would for an NVIDIA GPU.

## Step 0 — find out what you actually have

Providers differ in whether ROCm is preinstalled. SSH into the instance and run:

```bash
lspci | grep -i amd            # confirm an AMD GPU is present at all
rocminfo 2>/dev/null | grep -i "Marketing Name"   # GPU model, if ROCm is installed
rocm-smi 2>/dev/null           # driver + GPU status, if ROCm is installed
cat /opt/rocm/.info/version 2>/dev/null           # installed ROCm version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1
```

Three outcomes:
- `rocminfo`/`rocm-smi` work and `torch.cuda.is_available()` is already `True`
  → ROCm + a matching PyTorch build are preinstalled. Skip to Step 2.
- `lspci` shows an AMD Instinct card (MI100/MI210/MI250/MI300X) but no ROCm
  tools → bare image, do Step 1.
- The GPU is a consumer/RDNA card (Radeon RX series) → ROCm PyTorch support
  for consumer cards is much less consistently maintained than for Instinct
  (MI-series) cards. It may work, but check AMD's ROCm compatibility matrix
  for that specific card/ROCm-version combination before investing time —
  if it's not listed as supported, this is the point where I'd flag "can't
  set it up here" rather than push forward on an unsupported combination.

## Step 1 — install ROCm (only if not already present)

Follow AMD's official ROCm installation docs (rocm.docs.amd.com) for the
instance's exact OS/version — the package repo setup and required reboot
steps change often enough between ROCm releases that reproducing them here
risks being stale. In broad strokes: add AMD's ROCm apt repository for your
Ubuntu version, install the `rocm` (or `rocm-hip-sdk`) meta-package, add your
user to the `render` and `video` groups, then reboot. Re-run the Step 0
diagnostics afterward to confirm `rocm-smi` works before continuing.

## Step 2 — install PyTorch matching the installed ROCm version

**This is the one real gotcha, worth getting right the first time:** don't
run `pip install -r requirements.txt` as-is — its unpinned `torch>=2.2` will
pull the default PyPI build (CUDA/CPU), silently overwriting a working ROCm
build. `import torch` would still succeed either way, so the failure mode is
silent: `torch.cuda.is_available()` quietly goes back to `False` and training
falls back to CPU with no error message telling you why.

```bash
# 1. Check the installed ROCm version from Step 0 (e.g. 6.2), then install the
#    matching PyTorch ROCm wheel BEFORE installing the rest of requirements.txt:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2   # adjust the version tag to match

# 2. Verify GPU is actually visible before doing anything else:
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. NOW install the rest of the project's dependencies, explicitly skipping torch:
pip install transformers accelerate datasets huggingface_hub sentencepiece protobuf tqdm
```

If step 2's check doesn't print `True` and your GPU's name, stop and re-check
the ROCm/PyTorch version match before doing anything else — everything past
this point will silently run on CPU otherwise.

## Step 3 — clone the repo and run the same pipeline as Colab

From here it's identical to `docs/COLAB_SETUP.md` steps 3 onward — clone the
repo, checkout `main`, then:

```bash
scripts/build_llama_cpp.sh          # CPU-only build, no ROCm needed for this part
python scripts/download_model.py
python scripts/prepare_eval_data.py
scripts/convert_to_gguf.sh models/qwen3-0.6b-hf models/qwen3-0.6b-fp16.gguf
scripts/quantize_ptq.sh models/qwen3-0.6b-fp16.gguf models/qwen3-0.6b-ptq-q2_k.gguf

python scripts/train_qat.py \
  --base-model models/qwen3-0.6b-hf \
  --output-dir models/qwen3-0.6b-qat-hf \
  --max-steps 500 --save-every 100 --eval-every 50

scripts/export_qat_gguf.sh models/qwen3-0.6b-qat-hf
```

No `accelerate launch` needed — same as Colab, `Accelerator()` degrades
gracefully to single-device on one GPU, ROCm or CUDA alike.

## Things that differ from the Colab guide

- **No `google.colab.files.download()`.** Get the three `.gguf` files off the
  instance with `scp`/`rsync` back to the Mac mini, or upload to whatever
  storage the provider gives you (S3-compatible bucket, etc.).
- **No Drive-backed persistence.** If the instance is billed by the hour and
  you might tear it down, copy `models/*.gguf` off promptly once each step
  finishes rather than assuming the disk persists.
- **`--save-every` in `train_qat.py` still matters** — cloud GPU instances
  can still be preempted/interrupted depending on the provider's instance
  type (spot/preemptible pricing especially), same rationale as the Colab
  disconnect risk.
- **flash-attention isn't required** — this repo's training loop uses
  `transformers`' default attention path, not a `flash-attn` dependency
  (which has ROCm compatibility issues of its own via a separate fork). Not
  a concern here, just don't add flash-attn as an optimization later without
  checking ROCm support for it first.

## If it turns out not to work

If Step 0 shows a consumer RDNA card with no confirmed ROCm/PyTorch support,
or Step 2's PyTorch install can't find a matching ROCm wheel for the
installed driver version, that's the point to stop and fall back to Colab
(`docs/COLAB_SETUP.md`) instead — the two guides produce identical output
(`models/qwen3-0.6b-qat-hf`), so nothing downstream cares which one you used.
