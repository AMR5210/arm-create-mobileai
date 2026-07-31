#!/usr/bin/env python3
"""Export a QAT-trained HF checkpoint to GGUF Q2_K, in-process.

Replaces the old convert -> llama-quantize shell pipeline. Three deliberate
differences from letting llama-quantize re-derive everything:

  Fix 1 (scheme control): 2-bit tensors are packed with our own Q2_K encoder
    (qat/gguf_q2k.py, a faithful port of ggml's reference quantiser) so the
    export is transparent and controllable, rather than a black-box requant.
    Pair with `train_qat.py --group-size 16` so the QAT grouping lines up with
    Q2_K's 16-element sub-blocks.

  Fix 2 (mixed precision preserved): the layers kept full-precision during QAT
    (`--skip-layers`) are written as F16, not silently crushed to Q2_K like the
    old llama-quantize path did -- which had been throwing away the whole point
    of mixed-precision training.

  Fix 3 (tokenizer): the base model's tokenizer files are restored into the HF
    dir before conversion. QAT only trains weights, never the tokenizer, so
    restoring from the known-good base defends against any save_pretrained-time
    drift in tokenizer_config.json. This is a permanent, documented step, not a
    root-cause fix (a deliberate earlier decision).

Testing (needs llama.cpp binaries, so run on the box that has them):
  llama-perplexity -m <out.gguf> -f eval/data/wikitext2_test.txt
compared against the PTQ Q2_K baseline and the training-time proxy.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from qat.gguf_q2k import quantize_q2_k, roundtrip_error  # noqa: E402

# HF nn.Linear names -> GGUF tensor names.
_PROJ_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
# GGUF weight suffixes eligible for 2-bit (the set QAT fake-quantises).
_QUANTIZABLE = tuple(f"{v}.weight" for v in _PROJ_MAP.values())
# Standard tokenizer artifacts to restore from the base model (Fix 3).
_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
)


def skip_pattern_to_gguf(pattern: str) -> str | None:
    """Translate a training --skip-layers pattern (e.g.
    'layers.16.self_attn.k_proj') into a GGUF tensor name ('blk.16.attn_k.weight').
    Returns None if it doesn't look like a per-layer linear pattern."""
    if pattern.startswith("blk."):
        return pattern if pattern.endswith(".weight") else pattern + ".weight"
    m = re.search(r"layers\.(\d+)\.(.+)", pattern)
    if not m:
        return None
    layer, proj = m.group(1), m.group(2).rstrip(".")
    gguf_proj = _PROJ_MAP.get(proj)
    return f"blk.{layer}.{gguf_proj}.weight" if gguf_proj else None


def restore_tokenizer(hf_dir: Path, base_dir: Path) -> None:
    """Fix 3: copy the base model's tokenizer files into the QAT checkpoint dir,
    backing up any that differ to <name>.qat.bak. Safe because QAT never changes
    the tokenizer."""
    print(f"==> [Fix 3] Restoring tokenizer files from {base_dir}")
    restored = 0
    for fname in _TOKENIZER_FILES:
        src = base_dir / fname
        if not src.exists():
            continue
        dst = hf_dir / fname
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            shutil.copy2(dst, dst.with_suffix(dst.suffix + ".qat.bak"))
            print(f"    differs, backed up: {fname} -> {fname}.qat.bak")
        shutil.copy2(src, dst)
        restored += 1
    print(f"    restored {restored} tokenizer file(s)")


def convert_to_fp16(hf_dir: Path, fp16_out: Path, llama_dir: Path, force: bool) -> None:
    if fp16_out.exists() and not force:
        print(f"==> Reusing existing fp16 GGUF: {fp16_out}")
        return
    convert = llama_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        sys.exit(f"ERROR: {convert} not found. Clone llama.cpp into third_party/ first.")
    print(f"==> Converting {hf_dir} -> {fp16_out} (f16)")
    subprocess.run(
        [sys.executable, str(convert), str(hf_dir), "--outfile", str(fp16_out), "--outtype", "f16"],
        check=True,
    )


def rewrite_to_q2k(fp16_gguf: Path, out_gguf: Path, skip_gguf: set[str]) -> dict:
    """Read the fp16 GGUF, copy all metadata, and re-emit tensors: Q2_K for the
    quantizable non-skip linears, F16/F32 verbatim for everything else."""
    from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType, LlamaFileType

    reader = GGUFReader(fp16_gguf)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(str(out_gguf), arch)

    # Copy every KV field except the reader-managed header and file_type (which
    # we override to Q2_K below).
    for key, field in reader.fields.items():
        # Skip reader-managed header, the arch (already set by the writer ctor),
        # and file_type (overridden to Q2_K below).
        if key.startswith("GGUF.") or key in ("general.architecture", "general.file_type"):
            continue
        vtype = field.types[0]
        sub = field.types[1] if vtype == GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub)
    writer.add_key_value("general.file_type", int(LlamaFileType.MOSTLY_Q2_K), GGUFValueType.UINT32)

    counts = {"q2_k": 0, "f16_skip": 0, "copied": 0}
    sample_err = None
    for t in reader.tensors:
        name = t.name
        is_weight = name.endswith(".weight") and name.startswith("blk.")
        quantizable = is_weight and name.split(".weight")[0].endswith(
            tuple(v for v in _PROJ_MAP.values())
        )
        if quantizable and name in skip_gguf:
            writer.add_tensor(name, np.ascontiguousarray(t.data))  # F16 verbatim (Fix 2)
            counts["f16_skip"] += 1
        elif quantizable and t.data.ndim == 2 and t.data.shape[1] % 256 == 0:
            packed = quantize_q2_k(t.data.astype(np.float32))       # Q2_K (Fix 1)
            writer.add_tensor(name, packed, raw_dtype=GGMLQuantizationType.Q2_K)
            counts["q2_k"] += 1
            if sample_err is None and name.endswith("ffn_gate.weight"):
                sample_err = {"tensor": name, **roundtrip_error(t.data.astype(np.float32))}
        else:
            writer.add_tensor(name, np.ascontiguousarray(t.data))   # norms / embeddings / etc.
            counts["copied"] += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"counts": counts, "sample_roundtrip": sample_err}


def validate(out_gguf: Path, skip_gguf: set[str]) -> None:
    from gguf import GGUFReader, GGMLQuantizationType

    r = GGUFReader(out_gguf)
    types: dict[str, int] = {}
    for t in r.tensors:
        tn = GGMLQuantizationType(t.tensor_type).name
        types[tn] = types.get(tn, 0) + 1
    print(f"==> Validation: {len(r.tensors)} tensors; type histogram: {types}")
    missing = [s for s in skip_gguf if not any(t.name == s for t in r.tensors)]
    if missing:
        print(f"    WARNING: skip-layers not found in GGUF (name mismatch?): {missing}")
    for s in sorted(skip_gguf):
        t = next((x for x in r.tensors if x.name == s), None)
        if t is not None:
            kept = GGMLQuantizationType(t.tensor_type).name
            flag = "OK" if kept == "F16" else f"UNEXPECTED({kept})"
            print(f"    skip-layer {s}: {kept}  [{flag}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-dir", type=Path, default=Path("models/qwen3-0.6b-qat-hf"))
    p.add_argument("--base-model", type=Path, default=Path("models/qwen3-0.6b-hf"),
                   help="Source of known-good tokenizer files for the Fix 3 restore.")
    p.add_argument("--fp16", type=Path, default=Path("models/qwen3-0.6b-qat-fp16.gguf"))
    p.add_argument("--out", type=Path, default=Path("models/qwen3-0.6b-qat-q2_k.gguf"))
    p.add_argument("--skip-layers", nargs="+", default=[],
                   help="Same patterns passed to train_qat.py --skip-layers; kept F16 in the GGUF.")
    p.add_argument("--llama-dir", type=Path, default=Path("third_party/llama.cpp"))
    p.add_argument("--force-convert", action="store_true", help="Rebuild the fp16 GGUF even if present.")
    p.add_argument("--no-tokenizer-restore", action="store_true")
    args = p.parse_args()

    skip_gguf = set()
    for patt in args.skip_layers:
        g = skip_pattern_to_gguf(patt)
        if g is None:
            print(f"    [warn] could not map skip pattern to a GGUF tensor name: {patt}")
        else:
            skip_gguf.add(g)
    if skip_gguf:
        print(f"==> Skip-layers kept F16: {', '.join(sorted(skip_gguf))}")

    if not args.no_tokenizer_restore and args.base_model.exists():
        restore_tokenizer(args.hf_dir, args.base_model)

    convert_to_fp16(args.hf_dir, args.fp16, args.llama_dir, args.force_convert)

    print(f"==> [Fix 1/2] Packing Q2_K (+F16 skips) -> {args.out}")
    result = rewrite_to_q2k(args.fp16, args.out, skip_gguf)
    print(f"    tensors: {result['counts']}")
    if result["sample_roundtrip"]:
        sr = result["sample_roundtrip"]
        print(f"    round-trip self-check on {sr['tensor']}: "
              f"rel_mean_abs_err={sr['rel_mean_abs_err']:.4f} max_abs_err={sr['max_abs_err']:.4f}")

    validate(args.out, skip_gguf)
    print(f"\n==> Export complete: {args.out}")
    print("Next (on a box with llama.cpp built):")
    print(f"  llama-perplexity -m {args.out} -f eval/data/wikitext2_test.txt")


if __name__ == "__main__":
    main()
