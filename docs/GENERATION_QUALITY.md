# Generation quality across model variants

Compares output quality for the three benchmark variants under Qwen3's chat
template. Reproduce with `scripts/compare_generations.sh`.

Verify model identity with `scripts/verify_model_signatures.py` before running —
filenames alone do not establish which checkpoint is present.

## Models under test

| Variant | Size (bytes) | SHA-256 (first 16) | Source |
| --- | --- | --- | --- |
| `baseline-fp16` | 1,509,347,648 | `4d8b9bdc9fcf2fee` | local `convert_to_gguf.sh` |
| `ptq-2bit` | 479,777,024 | `31edf956a842de21` | local `quantize_ptq.sh` |
| `qat-2bit` | 495,193,952 | `a861b8924a2b1881` | HF `AMR5210/qwen3-0.6b-qat-q2k`, `fineweb-blend/` |

`qat-2bit` is the final FineWeb-blend checkpoint (step 48828), matching the
`qat_q2k_fineweb_final` signature: 187 Q2_K / 9 F16 skip-layers / 114 copied
tensors, with the 9-skip-layer set defined in
`results/wikitext2_perplexity.json`'s `common_training_config`.

## Evaluation protocol

Two properties of Qwen3-0.6B determine how it must be prompted:

1. **It is instruction-tuned and ships a chat template.** A base-style completion
   prefix does not exercise the model as intended.
2. **It is a reasoning model.** The template opens an assistant turn without a
   pre-closed `<think>` block unless `enable_thinking=false` is set, so the model
   spends its opening tokens reasoning. Short generation budgets return reasoning
   fragments rather than answers.

Three prompt formats are therefore evaluated:

| Mode | Description |
| --- | --- |
| `raw` | no chat template (base-style completion) |
| `think` | chat template, thinking enabled (Qwen3 default) |
| `nothink` | chat template, `enable_thinking=false` |

Both greedy and Qwen3's recommended sampling are exercised, as the Qwen3 model
card notes greedy decoding degrades reasoning models.

## Perplexity

Measured with `llama-perplexity`, `n_ctx=512` (583 WikiText-2 chunks, 550 C4
chunks).

| Variant | WikiText-2 | C4 |
| --- | --- | --- |
| `baseline-fp16` | 21.37 ± 0.19 | 29.46 ± 0.26 |
| `ptq-2bit` | 220.93 ± 2.34 | 279.30 ± 2.95 |
| `qat-2bit` | 18.46 ± 0.15 | 32.19 ± 0.28 |

QAT improves on PTQ by 11.97× on WikiText-2 and 8.68× on C4.

## fp16 baseline

Prompt: `"What is the capital of France? Answer in one sentence."`

| Mode | Output |
| --- | --- |
| `think` (60 tok) | `[Start thinking] Okay, the user is asking for the capital of France...` |
| `think` (320 tok) | closes the think block, answers `The capital of France is Paris.` |
| `nothink` (150 tok) | `The capital of France is Paris.` |

Reasoning consumes roughly the first 60 tokens in `think` mode, so the generation
budget must accommodate it. `nothink` produces a direct answer.

## PTQ 2-bit

Output is degenerate across every mode, sampler and prompt combination tested.

| Prompt | Output (`nothink`) |
| --- | --- |
| capital of France | `**الملك** **الملك** **الملك**...` |
| photosynthesis | `**الإسخاء** **الإسخاء**...` |
| 15 plus 27 | `**الإعداد** **الإعد** **الإعد**...` |
| three primary colors | `soh soh soh soh soh...` |

In `think` mode the output takes the form of English repetition loops
(`oy HALO HELLO HELLO HELTO...`, `So 15 and 27 is that. So 15 and 27 is that.`)
rather than an answer.

This is the expected quality collapse for naive 2-bit PTQ, and it holds under
correct prompting. The baseline carries F16 tied embeddings matching the QAT
export, so the result is not an artifact of an under-provisioned baseline.

## QAT 2-bit

`nothink`, 150 tokens, greedy — all four prompts answered correctly:

| Prompt | Output | Assessment |
| --- | --- | --- |
| capital of France | `The capital of France is Paris.` | correct; identical to fp16 |
| photosynthesis | `Okay, I need to write a sentence that explains what photosynthesis is. Let me think for a moment. Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into energy. It's a process that occurs in the chloroplasts of plant cells...` | correct, verbose |
| 15 plus 27 | `15 + 27 = 42 #15 + 27 = 42 So, 15 plus 27 is 42. Let me know if you have any other questions.` | correct |
| three primary colors | `...I think the three primary colors are red, blue, and yellow. But I'm not sure if that's the case. Let me check. Red, blue, and yellow are all primary colors. So, yeah...` | correct, hedged |

`think` mode at 320 tokens is likewise correct on all four, including
`15 + 27 = 42` and `The three primary colors are red, blue, and yellow.`

Raw completion at 16 tokens on-device returns `Paris, and the capital of Spain is
Madrid. The capital of Italy is Rome`, comparable to the fp16 baseline's
`Paris. The capital of Italy is Rome. The capital of Spain is Madrid.`

### Output characteristics

Relative to fp16, QAT output differs in presentation rather than correctness:

- **Verbosity.** Answers are restated several times and generation does not
  terminate cleanly, consistent with under-emission of EOS.
- **Reasoning preamble in `nothink` mode.** Output opens with `Okay, I need to...`
  or `Okay, let me try to...` even when the template pre-closes the think block.
- **Occasional stray tokens**, such as an unexpected emoji in `think` mode.

fp16 answers in a single sentence and stops. This presentation gap, rather than
answer accuracy, is the visible difference between the two.

## Recommendations for the side-by-side demo

- **Use `nothink` for all three variants.** It is the strongest mode for QAT —
  all four prompts correct, with a single-sentence answer on the France prompt —
  and also fp16's best. A single format across all variants keeps the comparison
  like-for-like.
- **The France prompt is the strongest single comparison**: fp16 and QAT return
  identical answers while PTQ produces a repetition loop.
- **Arithmetic is a strong demo case**: fp16 `42`, QAT `42`, PTQ repetition loop.
- **Include one longer answer** (photosynthesis or colors) so QAT's verbosity is
  represented alongside the one-line responses.
- **Cap generation length**, since QAT's verbosity is most visible in long,
  uncapped output.

The QAT-versus-PTQ gap is the result this comparison establishes. QAT remains
measurably below fp16 on presentation quality, and the write-up should say so.

## Instruction-following metric

The MMLU-style slice in `eval/data/instruction_eval.jsonl` (100 questions, four
subjects, four choices) does not separate these variants under any of three
scoring methods, so perplexity is the reported quality metric.

| Method | fp16 | ptq-2bit | qat-2bit | Chance |
| --- | --- | --- | --- | --- |
| Free-generation accuracy | 0.17 | 0.07 | 0.11 | 0.25 |
| Parse rate | 46% | 22% | 2% | — |
| **Forced choice over A/B/C/D** | **26%** | **19%** | **18%** | **25%** |

**Free generation** leaves 37–73% of answers without an extractable letter and
places every variant below chance, including the fp16 baseline
(`results/logs/instr_eval_diag_summary.json`). The figure therefore reflects the
extraction step rather than model quality.

**Parse rate** — whether free generation yields a well-formed letter at all —
separates the variants, but reflects output shape. QAT scores 2% because it opens
with an `Assistant:` role prefix and restates the instruction rather than
answering, leaving no letter inside the 8-token budget. A budget sweep on a
15-question stratified sample shows the size of that effect: QAT moves 6.7% → 20%
→ 13.3% at 8/32/64 tokens while fp16 holds at 46.7% throughout, and at 64 tokens
QAT's output turns to repetition rather than reaching an answer. Parse rate also
orders the variants opposite to perplexity, placing the strongest one last.

**Forced choice** scores the model's preference among exactly the four legal
answers, independent of output formatting and generation budget. Two
implementations are used: the iOS harness takes the argmax over the A/B/C/D logits
at the first generated position, and the desktop script uses a single-token GBNF
grammar (`root ::= [ABCD]`) with greedy sampling, which admits only those four
tokens and selects the highest-scoring one. Across all 100 questions on fp16 the
two agree on **100/100** predictions.

All three variants sit at or below the 25% chance line, with label bias
dominating the result:

| Variant | Prediction histogram | Accuracy |
| --- | --- | --- |
| fp16 | A 77, B 11, C 3, D 9 | 26% |
| ptq-2bit | A 73, B 1, D 26 | 19% |
| qat-2bit | **A 100** | 18% |

Gold answers are distributed A 18 / B 28 / C 30 / D 24, so a constant "A" response
scores 18%. `qat-2bit` selects A for all 100 questions and scores exactly that,
indicating no discrimination between choices. fp16's 26% is within noise of chance
alongside a 77% A-bias.

At this model size the slice carries no information about variant quality however
the answer is extracted. Forced-choice accuracy is recorded in each result file
for completeness, with parse rate and free-generation accuracy retained as
diagnostics, and is not reported alongside perplexity.

## Follow-ups

- **QAT verbosity and weak EOS.** Worth confirming the fine-tuning data terminated
  examples with `<|im_end|>`; run-on output across all prompts is consistent with
  that cause, as is the `Assistant:` prefix seen on the MCQ prompt.
- **QAT does not honour `enable_thinking=false`**, emitting a reasoning preamble
  regardless.

## On-device figures

`results/dev-ios-sim/` holds the five-metric records for all three variants,
measured on the iOS harness (simulator, CPU backend). Each records the SHA-256 of
the file it measured. `results/<tag>.json` remains reserved for physical-device
runs.

| Variant | Disk | Peak RAM | Prompt tok/s | Gen tok/s | Perplexity |
| --- | --- | --- | --- | --- | --- |
| `baseline-fp16` | 1509.3 MB | 3393.7 MB | 120.60 | 44.62 | 21.3706 |
| `ptq-2bit` | 479.8 MB | 2159.7 MB | 120.09 | 78.31 | 220.9065 |
| `qat-2bit` | 495.2 MB | 2188.8 MB | 118.40 | 83.33 | 18.4600 |

Perplexity reproduces the desktop reference for each variant to within 0.011%,
which is what establishes the on-device harness as correct. Throughput and RAM
are simulator figures on desktop-class silicon and are not reportable results.

## Changelog

- **2026-08-04** — Results regenerated against the final FineWeb-blend QAT
  checkpoint (step 48828). An earlier 12-skip-layer export was superseded.
- **2026-08-04** — PTQ baseline rebuilt with F16 tied embeddings for parity with
  the QAT export, and perplexity re-measured (WikiText-2 275.89 → 220.93, C4
  361.61 → 279.30). The change is a quantization-recipe fix; PTQ involves no
  training.
- **2026-08-05** — Five-metric on-device suite run for all three variants. The
  reported instruction metric became forced-choice accuracy over A/B/C/D,
  replacing parse rate and free-generation accuracy, both retained as
  diagnostics.
