#!/usr/bin/env bash
# Side-by-side generation quality check across the three model variants.
#
# A short raw-completion probe is not a representative read on an
# instruction-tuned model. Qwen3-0.6B ships a chat template and is a reasoning
# ("thinking") model: with `add_generation_prompt` and thinking enabled, the
# template opens an assistant turn with no pre-closed <think> block, so the model
# spends its first tokens reasoning. A small token budget then returns reasoning
# fragments rather than an answer.
#
# Each variant is therefore run three ways, separating prompt-format effects from
# quantization damage:
#   raw      -- no chat template (base-style completion)
#   think    -- chat template, thinking enabled (Qwen3 default)
#   nothink  -- chat template, enable_thinking=false (direct answer)
#
# Results inform the prompt format used for the "same prompt, three models" demo
# described in docs/METHODOLOGY.md. See docs/GENERATION_QUALITY.md.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$ROOT_DIR/third_party/llama.cpp/build/bin/llama-cli"

N_PREDICT="${N_PREDICT:-200}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/generation_compare}"
PROMPT="${PROMPT:-What is the capital of France? Answer in one sentence.}"

# Sampling. Greedy matches the reported metrics; Qwen3's model card notes greedy
# decoding degrades reasoning models, so both are exercised here.
GREEDY=(--temp 0 --top-k 1)
QWEN_THINK=(--temp 0.6 --top-p 0.95 --top-k 20)
QWEN_NOTHINK=(--temp 0.7 --top-p 0.8 --top-k 20)

if [ ! -x "$CLI" ]; then
  echo "ERROR: $CLI not found. Run scripts/build_llama_cpp.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

declare -a VARIANTS=(
  "baseline-fp16:models/qwen3-0.6b-fp16.gguf"
  "ptq-2bit:models/qwen3-0.6b-ptq-q2_k.gguf"
  "qat-2bit:models/qwen3-0.6b-qat-q2_k.gguf"
)

run_one() {
  local tag="$1" model="$2" mode="$3" sampler_name="$4"; shift 4
  local sampler=("$@")
  local out="$OUT_DIR/${tag}.${mode}.${sampler_name}.txt"

  local -a args=(-m "$ROOT_DIR/$model" -n "$N_PREDICT" --no-warmup -st)
  case "$mode" in
    raw)     args+=(-no-cnv -p "$PROMPT") ;;
    think)   args+=(--jinja -p "$PROMPT") ;;
    nothink) args+=(--jinja --chat-template-kwargs '{"enable_thinking":false}' -p "$PROMPT") ;;
  esac
  args+=("${sampler[@]}")

  echo "  [$tag / $mode / $sampler_name]"
  "$CLI" "${args[@]}" </dev/null >"$out" 2>"$out.stderr"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    (llama-cli exited $rc -- see $out.stderr)"
  fi
}

echo "==> prompt: $PROMPT"
echo "==> n_predict: $N_PREDICT"
echo

for entry in "${VARIANTS[@]}"; do
  tag="${entry%%:*}"; model="${entry#*:}"
  if [ ! -f "$ROOT_DIR/$model" ]; then
    echo "  [$tag] SKIP -- $model not found"
    continue
  fi
  run_one "$tag" "$model" raw     greedy "${GREEDY[@]}"
  run_one "$tag" "$model" think   greedy "${GREEDY[@]}"
  run_one "$tag" "$model" nothink greedy "${GREEDY[@]}"
  run_one "$tag" "$model" think   qwen   "${QWEN_THINK[@]}"
  run_one "$tag" "$model" nothink qwen   "${QWEN_NOTHINK[@]}"
done

echo
echo "==> outputs in $OUT_DIR"
