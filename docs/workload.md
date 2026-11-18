# First P150a workload

The first release target is one P150a running Llama-3.1-8B-Instruct for one
user, with greedy text completion. SmolLM-135M is the smaller development
fixture; passing its gates doesn't qualify Llama. This page fixes the inputs and
the gates, as specified on October 29, 2026. Each gate's result on the card goes
in the [acceptance record](https://github.com/0xalkinun/baretile/issues/5), as
passed, failed or not run, with its command, log and source revision.

## Fixed inputs

| Input | Revision |
| --- | --- |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M/tree/1d461723eec654e65efdc40cf49301c89c0c92f4) | `1d461723eec654e65efdc40cf49301c89c0c92f4` |
| [Tenstorrent reference tokens](https://github.com/tenstorrent/tt-metal/blob/9f98dee9c5a4227b3fd866aa6662e02ba2b362af/models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt) and book text | tt-metal `9f98dee9c5a4227b3fd866aa6662e02ba2b362af` |

[workload.sha256](workload.sha256) holds the checksum of every file. Download
them into clean folders outside the checkout, with `huggingface_hub`'s `hf` and
a login that can read the gated Llama, and stop if a checksum fails:

```sh
export WORKLOAD_DIR=/absolute/path/to/workload-files
export LLAMA_DIR="$WORKLOAD_DIR/Llama-3.1-8B-Instruct"
export SMOLLM_DIR="$WORKLOAD_DIR/SmolLM-135M"
export REF_DIR="$WORKLOAD_DIR/reference"
mkdir -p "$REF_DIR"

hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --local-dir "$LLAMA_DIR" \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json model.safetensors.index.json \
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
hf download HuggingFaceTB/SmolLM-135M \
  --revision 1d461723eec654e65efdc40cf49301c89c0c92f4 \
  --local-dir "$SMOLLM_DIR" \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json model.safetensors

TT_REFERENCE_URL=https://raw.githubusercontent.com/tenstorrent/tt-metal/9f98dee9c5a4227b3fd866aa6662e02ba2b362af/models/tt_transformers/tests
curl --fail --location --max-time 120 \
  "$TT_REFERENCE_URL/reference_outputs/Llama-3.1-8B-Instruct.refpt" \
  --output "$REF_DIR/Llama-3.1-8B-Instruct.refpt"
curl --fail --location --max-time 120 \
  "$TT_REFERENCE_URL/tale-of-two-cities.txt.bz2" \
  --output "$REF_DIR/tale-of-two-cities.txt.bz2"

(cd "$WORKLOAD_DIR" && sha256sum --check) < docs/workload.sha256
```

An extra `chat_template.jinja` or an edited generation config changes the
input. Pass these folders to the commands; the server's model-ID download
doesn't pin a revision.

The reference holds 1,024 book tokens and their next-token predictions. The
prompt is its first 512 tokens verbatim, its BOS included, with no chat
template, and scoring starts at position 511. The autonomous and HTTP gates use
these prompts, in UTF-8, without BOS, a chat template or a trailing newline:

```text
The meaning of life is
The capital of France is
1, 2, 3,
```

## Limits

All layers and the full vocabulary run on the card, with no layer cutting,
vocabulary reduction or CPU fallback; tokenizing and sampling run on the host.
Llama runs in the `tt` precision mode, which
[`llama.precision`](../baretile/models/llama.py) defines, with a bfp8 KV cache;
SmolLM and the diagnostics run in `bf16`. Batch size is 1, in one slot of 1,024
positions, with 512 prompt tokens and 32 output tokens. Greedy means
temperature 0, top_k 0, top_p 1, min_p 0, no penalties and no speculative
decoding.

## Correctness gates

A skip, timeout, nonfinite logit or incomplete run is not a pass. A threshold
changes only through a reviewed change to this page, with the failing evidence.

| Gate | Required result | Command |
| --- | --- | --- |
| Tokenizer | The same token IDs and text as Hugging Face's tokenizer from the same folder | `python3 test/tokenizer.py "$SMOLLM_DIR" "$LLAMA_DIR"` |
| SmolLM checkpoint | `test_llama_model` passes | `LLAMA="$SMOLLM_DIR" python3 -m pytest -s test/test_models.py::test_llama_model` |
| SmolLM generation, `bf16` | 3 prompts × 32 steps on its own picks: logits within 3% of transformers', and each pick within 1% of the span below the best | `python3 test/generation.py "$SMOLLM_DIR" smollm.json 32 bf16` |
| Llama reference agreement, `tt` | At least 450/500 top-1 and 490/500 top-5 matches | `python3 -m baretile.models.llama "$LLAMA_DIR" "$REF_DIR/Llama-3.1-8B-Instruct.refpt" 500 tt` |
| Llama generation, `tt` | 3 prompts × 32 steps on its own picks, each pick in a bf16 transformers reference's top five | `python3 test/generation.py "$LLAMA_DIR" llama.json 32 tt` |
| Single-slot HTTP | The three prompts, streamed and not, give the same greedy text and usage, and a cancelled stream gives its slot back | `python3 test/workload.py http "$LLAMA_DIR" http.json` |

The scripts' opening comments give each gate in full. The reference agreement
floors are the pinned vendor's
[P150 targets](https://github.com/tenstorrent/tt-metal/blob/9f98dee9c5a4227b3fd866aa6662e02ba2b362af/models/model_targets.yaml#L685).
`llama`'s command prints the percentages without failing on a low score;
[bring-up](bring-up.md)'s llama stage runs it and fails below the gate. The same
command with `bf16`, and with `tt numpy`, which runs on the host, tells weight
precision from device arithmetic. On the host, `tt numpy` meets the gate with no
top-5 margin, 458 and 490 of 500;
[#47](https://github.com/0xalkinun/baretile/issues/47) has the fallback
precision.

`python3 test/workload.py time "$LLAMA_DIR" "$REF_DIR/Llama-3.1-8B-Instruct.refpt" timing.json`
runs the timing protocol that its opening comment describes, in three fresh
processes, and records the device, the host, the package versions, the model
files' hashes, baretile's commit and the command. There's no tokens/s floor
yet. CI runs these scripts on a small random Llama (`test_autonomous_generation`
and `test_workload_commands`), not on these checkpoints.
