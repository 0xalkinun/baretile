# Models and serving

The code in [baretile/models/](../baretile/models) runs pretrained models on
baretile and serves them over HTTP. It's layer 4: it uses `Tensor` and adds the
kernels that only the models need. [Adapters](adapters.md) runs tinygrad and
PyTorch programs, [training](training.md) trains GPT-2, and the
[first P150a workload](workload.md) pins a model and the gates it must pass.

## Running a model

`baretile/models/gpt2.py` and `baretile/models/llama.py` can run the same model
operations with baretile tensors or NumPy arrays. The generation commands
compare the two at each step. They feed NumPy's chosen token to both versions on
the next step, so one disagreement won't make the rest of the comparison
useless.

`llama.load` reads a local Hugging Face config and safetensors weights. It runs
these families, by the `model_type` in `config.json`, or in its `text_config`
for a model that also sees images:

| Family | `model_type` | For example | Checked on ttsim |
| --- | --- | --- | --- |
| Llama 3, SmolLM | `llama` | Llama-3.1-8B-Instruct, SmolLM-135M | Against transformers |
| Mistral | `mistral` | Mistral-7B | Against transformers, with its sliding window |
| Mixtral | `mixtral` | Mixtral-8x7B | Against transformers |
| Qwen2, Qwen2.5 | `qwen2` | Qwen2.5-0.5B | Against transformers |
| Qwen2-MoE | `qwen2_moe` | Qwen1.5-MoE-A2.7B | Against transformers |
| Qwen3 | `qwen3` | Qwen3-0.6B | Against transformers |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B | Against transformers |
| Qwen3.5, Qwen3.6 | `qwen3_5_moe_text`, `qwen3_5_text` | Qwen3.6-35B-A3B | The mixtures of experts against transformers; the dense models not compared |
| gpt-oss | `gpt_oss` | gpt-oss-20b | Against transformers |

The comparisons use small models with random weights, in `test_hf_families`,
`test_qwen3_moe` and `test_qwen3_5_moe`. They need PyTorch and transformers,
and skip without them. SmolLM-135M's real checkpoint is checked against NumPy
when `LLAMA` names it (`test_llama_model`). Another `model_type` fails at load
with a `ValueError` that names it, since a family such as Gemma has weights of
the same names but computes differently. So does a config with `mlp_bias`, since
the model adds no MLP biases. A config without `rope_theta` or
`original_max_position_embeddings` takes transformers' default.

The tokenizer reads byte-level BPE data from `tokenizer.json`. It uses the
file's merge order and `ignore_merges` setting; files without a merge list use
token IDs as ranks, as in tiktoken. Split, Digits and ByteLevel pre-tokenizers
run in their configured order. Loading raises `ValueError` for BPE dropout,
subword affixes, byte fallback, an unknown token, ByteLevel's prefix space or a
normalizer other than NFC. So the SentencePiece tokenizers of Mistral-7B and
Mixtral-8x7B don't load, and their checkpoints run only from Python, with token
ids of your own. The standalone GPT-2 command also needs `tiktoken`. Import the
models as `from baretile.models import llama`; they need NumPy, which
`pip install 'baretile[models]'` installs.

Choose the precision as the fourth argument after the script name:

```sh
python3 -m baretile.models.llama /path/to/SmolLM-135M "Hello" 6 bfp8
python3 -m baretile.models.gpt2 /path/to/gpt2/model.safetensors "Hello" 6 bf16
```

`bf16` is the default. `bfp8` stores weight matrices in bfp8. The Llama `bfp4`
mode puts gate and up projections in bfp4 and the other matrices in bfp8.
`tt` follows Tenstorrent's performance configuration, including its fidelity
settings. `experts` puts the MLP matrices in bfp4, the other matrices in bfp8,
and the KV cache in bfp8. `llama.precision` has the settings for each operation.
We haven't measured Qwen3.6 token accuracy with all matrix weights in bfp8 or
with a bfp4 KV cache.

In every mode, the Llama families keep the residual stream between blocks in
fp32. The norm after each half of a block adds the half's output, its last
matmul's fp32 sum, to the stream and reads the sum rounded to bf16. In bf16, a
large activation that a later block cancels would lose the low bits of what's
left: SmolLM-135M has one of about 31,000, which a bf16 copy holds only to
within 128. GPT-2's stream is still bf16. With a few rows, as in a decode or
verify step, the add and the norm are one launch, `llama.rms_kernel`: each row
of the grid takes a tile row of the stream, and its workers send each other
their sums of squares over NoC 1. The grid's other workers add the tiles, write
the sum out and send each tile to the worker that norms it.

Qwen3 and Qwen3.5 RMS-norm each head of q and k before RoPE. The norm reads the
qkv matmul's fp32 sum, runs in fp32 and rounds once to bf16 for RoPE. In bf16,
the norm's scale would round once for a whole head, and a score doesn't average
that error out over its products, as it does the roundings of single values.
Qwen3-0.6B's norm weights make scores near 48, where a rounding of 2^-8 is 0.19.
On the chip the norm and RoPE are one launch, `llama.rope_kernel`, a head of a
tile row a unit. The swap of a head's halves is a choice of tile, where the
tensor ops take a matmul by `turn`. It adds the two products exactly before it
rounds, where the FPU's add aligns the smaller to the larger's 10 bits first, so
the two can differ by a bf16 step in the dimensions that turn.

The loader fuses q, k and v projections, and gate and up projections, where
appropriate. Matmul weights are stored by tile columns. The KV cache is
allocated once and zeroed on the chip, and each token writes its keys and values
at its position.

## Experts and Gated DeltaNet

`baretile/models/qwen.py` contains the operations that extend a Llama block.
Expert routing uses `topk` on the chip. For decode steps and chunks of up to 128
tokens, `touched` lists the selected experts so tokens can share their weight
reads. Above that, it takes whichever of `touched` and `group`, which sorts
tokens by expert, has fewer tile rows. Routing holds a prompt chunk's choices in
one worker's L1, so it raises `ValueError` for a chunk of several thousand
tokens. The server's chunks of 2,048 tokens fit. From Python,
`serve.Engine(chunk=...)` and `llama.prefill`'s `tokens` set the chunk.

Qwen3.5 and Qwen3.6 combine attention with Gated DeltaNet. A DeltaNet block
keeps convolution history and a fixed-size recurrent state. Its decode kernel
performs the convolution, gates and state update in fp32. DeltaNet needs key and
value head dimensions that are multiples of 32, and a convolution of 4 taps,
which `llama.config` checks.

Prompt processing works in pieces of 64 tokens. Within a piece, each token's
correction solves a triangular system, whose inverse `qwen.corrected` builds by
joining blocks, from single tokens up. A product of powers over the whole piece
is the same inverse, but where a piece's keys point nearly the same way its
powers reach 1e10 and float32 loses the result. On the chip a piece is one
kernel, `qwen.rule`: every product on the FPU, which reads the fp32 tiles as
TF32, and the rest on the SFPU. State updates are written in place so later
replays see the new state.

These operations also take tokens in an order of their own, such as a tree's:

- `qwen.conv` is the convolution in one launch, over picked rows of a table.
  With `last`, a pick past the table's rows reads the conv state where it lies.
- `qwen.conv_state` writes the rows that its picks name into the conv state, in
  one launch.
- `qwen.rule` runs a piece's rule. For a piece of one tile, its kernel,
  `rule_kernel`, takes q and k to length 1 itself, and with `raw` also makes the
  gates and gives its outputs RMS-normed and gated, as a DeltaNet block's last
  step.
- `qwen.carry` gives the state after a path through a piece that `rule` ran, as
  a prompt of the path would leave it, in one launch.
- A `rule` of the caller's takes the place of the convolution and the
  recurrence in `qwen.deltanet`, and leaves the states as they were.

Models with a multi-token prediction layer can draft a token and verify it with
the next token. Verification keeps state after the first token and after both
tokens. The next step reads whichever state matches the accepted tokens. Free
server slots leave their recurrent states unchanged. `qwen.predict` is the
prediction layer's pass on its own, with the caller's rows, positions and mask,
and `llama.load(mtp=...)` takes that layer's weights from a separate file. A
hybrid cut down to its recurrent blocks also runs: with no attention cache to
give the positions, `llama.forward` takes them from its `capacity` argument or
the RoPE tables.

## Serving

The server accepts a folder or Hugging Face model ID. Its positional arguments
are model, port, slots, positions (`places` in its usage line), precision,
optional KV format and optional connection limit:

```sh
python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000 4 512 bf16
```

Here, four requests can run at once, with 512 positions per slot. Choose a
multiple of 32 that the model's RoPE tables cover, and leave room for the answer
as well as the prompt. A prompt can have up to positions − 2 tokens: the engine
reserves the last position of each slot for steps where the slot is idle. Port
0 picks a free port, which the server prints when it's ready.

Downloads go to `$XDG_CACHE_HOME/baretile/hub`, `~/.cache/baretile/hub` by
default, and are reused. `HF_TOKEN` supplies access to a gated model, and
`HF_ENDPOINT` changes the download endpoint. A redirect to another origin
doesn't get the token. The server binds to all interfaces on the selected port
and has no API key check.

It serves `GET /v1/models`, `POST /v1/completions` and
`POST /v1/chat/completions`. Set `stream` to true to receive tokens as they're
generated, and `stream_options.include_usage` to also get the token counts in a
last chunk whose `choices` is empty. You can set temperature, top_k, top_p,
min_p, seed, stop strings and repetition penalties. Defaults come from the
model's `generation_config.json` when present. `max_tokens` must be a positive
integer; a chat request can name it `max_completion_tokens`, which takes
precedence. A message's content is a string or a list of text parts. `top_k=0`
keeps all tokens. Temperature must be nonnegative, `top_p` in `(0, 1]` and
`min_p` in `[0, 1]`. A nonzero temperature must be a normal fp32 value, and a
repetition penalty a positive one; presence and frequency penalties must fit
fp32. Invalid requests get HTTP 400 before they take a slot.

A connection has 10 seconds to send its headers and body, or the server closes
it, with HTTP 408 once its request line has arrived. The body needs one
`Content-Length` and no chunked transfer encoding (HTTP 400), and at most 8 MiB
(HTTP 413). By default at most 4 × (slots + 8) connections have handler
threads, 48 for four slots, and one client address may hold a quarter of them.
Past the total a connection gets HTTP 503, and past its address's share HTTP
429, each with a JSON error that names the limit. The seventh argument sets the
total, for example
`python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000 4 512 bf16 "" 64`,
where `""` keeps the precision's KV format. Behind a reverse proxy every client
has the proxy's address. To give that address all of them, start the server
from Python as `serve.main` does, and pass `per_client` equal to `connections`
to `serve.server`. It also takes `input_wait` and `body_limit`.

EOS tokens and sampling defaults come from `generation_config.json`, then from
`text_config` in `config.json`, then from the rest of `config.json`, field by
field. Where none of them has an EOS, `tokenizer_config.json`'s `eos_token` is
the EOS when it's one token. An explicit null EOS means no automatic end token.

The engine gives each request a slot in a shared KV cache and decodes occupied
slots together. Prompts are processed in chunks between decode steps. A request
gets HTTP 429 when no slot is free. A non-streaming request that receives no
text for 600 seconds gets HTTP 504, and a stream ends with a chunk whose
`finish_reason` is `error`, then `[DONE]`. A request that times out, or whose
client hangs up, is cancelled, and the engine releases its slot. A failure in a
request's own work on the host, such as sampling, ends that request with an
error and frees its slot. A failure on the chip, including reading logits back,
stops the engine: pending requests end with an error and new requests get HTTP
503.

## Reusing a conversation

When a prompt arrives, the engine looks for a free slot that has already
processed a prefix of it. It can also restore a saved session with its tokens,
KV cache and recurrent state. DeltaNet must resume from the exact tokens that
produced its state.

Sessions are saved in spare DRAM on the chip. To make room for a new one, the
server moves the least recently used sessions to host memory, which has a size
limit too, past which the oldest sessions are dropped. Restoring a session from
host memory uploads all of it before it changes the slot; if DRAM can't hold it,
the server drops the session and prefills the prompt from the start. With
speculation, a session includes the prediction layer's cache rows for its
tokens. The response's usage field counts the prompt tokens that were reused.

## Sampling and API limits

For supported vocabulary shapes, the device returns the largest 64 logits and
the normalization sum, which greedy selection and top_k up to 64 can use. A
greedy pick of 4 rows or more takes `serve.grouped_topk`, which reads only the
groups of a row's columns that can hold its largest. `serve.frontier` picks a
step of a draft tree grown best first on the chip and logs its picks and
candidates, so a round reads its tree back once. Penalties, logprobs and
constrained output need the full row, and so do some other sampling requests.
When a stop string cuts the text, logprobs cover only the returned text's
tokens.

Chat templates are Python approximations of each model's template, so a model
with a new template may need code in `serve.py`. A `chat_template` that is a
list of named templates uses the one named `default`. gpt-oss's refuses a
system, developer or tool message after the first message with HTTP 400, and
takes a `reasoning_effort` of `low`, `medium` or `high`. Qwen tool calls are
returned in `tool_calls`, and reasoning goes in `reasoning_content`. Each tool
needs a function with a nonempty name, and its `parameters`, if not null, an
object schema. A stream holds back text that may begin a reasoning or tool-call
marker until the marker is complete or ruled out. `test_chat_tools`,
`test_serve_http`, `test_serve_schema` and `test_serve_chat` have example
requests to start from when adding a model or client.

`tool_choice` and JSON response formats constrain which tokens can be sampled.
JSON schemas support types, properties, required fields, additional properties,
arrays, string lengths, `pattern`, numeric bounds and `multipleOf`, enums,
constants, alternatives, `allOf` and nonrecursive local references. Annotations
such as `title`, `description` and `format` change nothing. Any other keyword,
such as `not`, `prefixItems` or `uniqueItems`, gets HTTP 400 naming it, and so
do patterns with lookarounds, backreferences, `\b`, `\p` or `\c`, and a
schema's patterns past `serve.PATTERN_STATES` NFA states together. Arrays and
objects nest at most 32 deep (`serve.NESTING`), and so do schemas. If no token
can continue a constrained answer, or it stops before it's complete, the request
ends with an error: HTTP 500, or `finish_reason` `error` in a stream.
`serve.formatted` documents the supported subset.

## Estimating cost

`baretile/models/cost.py` estimates memory use and serving cost from a model
config:

```sh
python3 -m baretile.models.cost /path/to/Qwen3.6-35B-A3B experts context=16384 accept=0.55
```

`probe=record/probe.json` takes a card's rates from the probe's record, which
holds the AI clock it measured at. Without one, the model assumes that decode
gets 70% of the specified 512 GB/s of DRAM bandwidth and that a launch takes one
microsecond. Compute uses the fidelity phases that the precision mode selects
for each operation; `phases=number` overrides them all, and the report says so.
Memory estimates reserve the full context for each attention cache, as the
allocator does. If the weights leave no room for a session, the report says so
and omits throughput.

The model counts the weights, caches, states and activations that a step reads.
It estimates how long those reads and the matrix operations take, uses the
longer time, then adds the cost of launches and RISC-V control code. The number
of different experts selected and the fraction of drafts accepted will change
with your workload. [Training](training.md#estimating-a-run) has the estimate
for a training run.
