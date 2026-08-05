#!/usr/bin/env python3
"""Copy a GGUF file, omitting named tensors.

Used to make the PTQ baseline structurally match the QAT export. Qwen3-0.6B sets
`tie_word_embeddings: true`, but `convert_hf_to_gguf.py` materializes
`output.weight` as a byte-identical duplicate of `token_embd.weight`. The QAT
exporter omits it and relies on llama.cpp's tied fallback, so without this step
the two 2-bit models differ by one 311 MB tensor and ~0.16 B reported parameters
-- which would undercut the "same footprint, only quality differs" premise of
Claim B in docs/METHODOLOGY.md.

`llama-quantize` can change tensor types but not remove a tensor, so dropping one
requires a separate rewrite pass. Removing a byte-identical duplicate is
numerically lossless: llama.cpp falls back to `token_embd.weight` for the output
projection when `output.weight` is absent.

The field-copy logic follows gguf-py's own gguf_new_metadata.py.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402

logger = logging.getLogger("gguf_drop_tensor")


def copy_without(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, drop: set[str]) -> list[str]:
    for field in reader.fields.values():
        # Architecture and GGUF.* are written by GGUFWriter itself.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if value is not None:
            writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    kept, dropped = [], []
    for tensor in reader.tensors:
        if tensor.name in drop:
            dropped.append(tensor.name)
            continue
        kept.append(tensor)
        writer.add_tensor_info(
            tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in kept:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--drop", action="append", required=True,
                    help="tensor name to omit; repeatable")
    ap.add_argument("--require-duplicate-of", metavar="TENSOR",
                    help="require every --drop tensor to be byte-identical to TENSOR. "
                         "Guards against discarding weights that carry unique values.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reader = gguf.GGUFReader(args.input)
    names = {t.name for t in reader.tensors}
    drop = set(args.drop)

    missing = drop - names
    if missing:
        sys.exit(f"ERROR: tensor(s) not present in {args.input}: {sorted(missing)}")

    if args.require_duplicate_of:
        ref = [t for t in reader.tensors if t.name == args.require_duplicate_of]
        if not ref:
            sys.exit(f"ERROR: reference tensor {args.require_duplicate_of} not in {args.input}")
        import numpy as np

        ref_bytes = np.asarray(ref[0].data).view(np.uint8).ravel()
        for name in sorted(drop):
            t = [x for x in reader.tensors if x.name == name][0]
            b = np.asarray(t.data).view(np.uint8).ravel()
            if b.shape != ref_bytes.shape or not bool((b == ref_bytes).all()):
                sys.exit(
                    f"ERROR: {name} is not byte-identical to {args.require_duplicate_of}. "
                    "Dropping it would discard unique weights; stopping."
                )
            logger.info(f"{name} is byte-identical to {args.require_duplicate_of}")

    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(args.output, arch, endianess=reader.endianess)
    dropped = copy_without(reader, writer, drop)

    logger.info(f"dropped {len(dropped)} tensor(s): {dropped}")
    logger.info(f"wrote {args.output} "
                f"({args.output.stat().st_size / 1e6:.1f} MB, "
                f"{len(reader.tensors) - len(dropped)} tensors)")


if __name__ == "__main__":
    main()
