#!/usr/bin/env python3
"""Verify that models/ contains the exact files the recorded results describe.

Run before a benchmark session, and before citing any figure one produces.

A GGUF's filename does not establish its identity. Two exports of the same model
can share a name and an approximate size, load without error and generate
plausible text, while differing in skip-layer selection or quantization recipe --
and therefore describe different checkpoints. Checking size, tensor mix,
skip-layer set and SHA-256 against a recorded signature makes the identity
explicit.

Expected signatures live in scripts/model_signatures.json.

Exit codes: 0 all checks passed, 1 mismatch, 2 setup problem.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

try:
    import numpy as np
    from gguf import GGUFReader
except ImportError as e:  # pragma: no cover
    sys.exit(f"setup error: {e}. Is third_party/llama.cpp cloned and numpy installed?")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Checker:
    def __init__(self, use_color: bool = True):
        self.failures: list[str] = []
        self.checks = 0
        self.c = use_color

    def _paint(self, s: str, colour: str) -> str:
        return f"{colour}{s}{RESET}" if self.c else s

    def check(self, label: str, got, want) -> bool:
        self.checks += 1
        ok = got == want
        if ok:
            print(f"    {self._paint('PASS', GREEN)}  {label}")
        else:
            print(f"    {self._paint('FAIL', RED)}  {label}")
            print(f"            got:  {got}")
            print(f"            want: {want}")
            self.failures.append(label)
        return ok


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_gguf_facts(path: Path) -> dict:
    r = GGUFReader(path)
    ts = r.tensors
    names = {t.name for t in ts}
    return {
        "n_tensors": len(ts),
        "tensor_mix": dict(collections.Counter(t.tensor_type.name for t in ts)),
        "names": names,
        "n_elements": sum(int(np.prod(t.shape)) for t in ts),
        "token_embd_type": next(
            (t.tensor_type.name for t in ts if t.name == "token_embd.weight"), None
        ),
        "output_weight_present": "output.weight" in names,
        "f16_non_embd": sorted(
            t.name for t in ts if t.tensor_type.name == "F16" and t.name != "token_embd.weight"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "scripts" / "model_signatures.json")
    ap.add_argument("--skip-hash", action="store_true",
                    help="skip sha256 (fast structural check only)")
    ap.add_argument("--only", action="append", help="check only this variant; repeatable")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"setup error: manifest {args.manifest} not found", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text())
    ck = Checker(use_color=not args.no_color)
    facts: dict[str, dict] = {}

    for tag, spec in manifest["models"].items():
        if args.only and tag not in args.only:
            continue
        path = ROOT / spec["path"]
        print(f"\n=== {tag}  ({spec['path']})")
        if not path.exists():
            print(f"    {ck._paint('FAIL', RED)}  file missing")
            ck.failures.append(f"{tag}: missing")
            continue

        print(f"    {DIM if not args.no_color else ''}provenance: {spec['provenance']}"
              f"{RESET if not args.no_color else ''}")
        if spec.get("recorded_run"):
            print(f"    {DIM if not args.no_color else ''}recorded:   {spec['recorded_run']}"
                  f"{RESET if not args.no_color else ''}")

        ck.check(f"{tag} size_bytes", path.stat().st_size, spec["size_bytes"])
        if not args.skip_hash and spec.get("sha256"):
            ck.check(f"{tag} sha256", sha256(path), spec["sha256"])

        f = load_gguf_facts(path)
        facts[tag] = f
        ck.check(f"{tag} n_tensors", f["n_tensors"], spec["n_tensors"])
        ck.check(f"{tag} tensor_mix", f["tensor_mix"], spec["tensor_mix"])
        ck.check(f"{tag} token_embd type", f["token_embd_type"], spec["token_embd_type"])
        ck.check(f"{tag} output.weight present",
                 f["output_weight_present"], spec["output_weight_present"])
        if spec.get("skip_layers") is not None:
            ck.check(f"{tag} skip-layer set", f["f16_non_embd"], sorted(spec["skip_layers"]))

    par = manifest.get("parity")
    if par and all(t in facts for t in par["between"]):
        a, b = par["between"]
        fa, fb = facts[a], facts[b]
        print(f"\n=== parity: {a} <-> {b}")
        ck.check("both n_tensors", (fa["n_tensors"], fb["n_tensors"]),
                 (par["n_tensors"], par["n_tensors"]))
        ck.check("both n_elements", (fa["n_elements"], fb["n_elements"]),
                 (par["n_elements"], par["n_elements"]))
        ck.check("identical tensor names", fa["names"] == fb["names"],
                 par["identical_tensor_names"])
        ck.check("both tied", (not fa["output_weight_present"], not fb["output_weight_present"]),
                 (par["both_tied"], par["both_tied"]))
        ck.check("shared token_embd type", (fa["token_embd_type"], fb["token_embd_type"]),
                 (par["shared_token_embd_type"], par["shared_token_embd_type"]))
        print(f"    {YELLOW if not args.no_color else ''}known divergence:{RESET if not args.no_color else ''} "
              f"{par['known_divergence']}")

    print("\n" + "=" * 72)
    if ck.failures:
        print(ck._paint(f"FAILED: {len(ck.failures)} of {ck.checks} checks", RED))
        for f in ck.failures:
            print(f"  - {f}")
        print("\nResolve these before citing benchmark figures from these files.")
        return 1
    print(ck._paint(f"OK: all {ck.checks} checks passed", GREEN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
