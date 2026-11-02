# Changelog

baretile is pre-alpha. Everything below is tested on Tenstorrent's ttsim
simulator. A P150a has run the first bring-up stages, preflight, chip and
transfers; nothing else is tested on a card yet.

## 0.0.7

The release from a pass over the whole repository: fixes to wrong results in reductions, matmuls, slices and block
floats, configs and requests that are refused where they used to compute something else, and the package's links at
the repository's new home.

### Changed

- The repository moved to [github.com/0xalkinun/baretile](https://github.com/0xalkinun/baretile), and the package's
  project links point there.
- Two `Chip`s on one device object raise `RuntimeError`, as a second opener of the device does, since they'd hand out
  the same TLBs and DRAM.
- The server refuses, with HTTP 400 naming them, JSON-schema keywords it doesn't know, such as `not`, `prefixItems`
  and `uniqueItems`, where it ignored them. It enforces `allOf`, and accepts annotations like `title`.
- gpt-oss chats refuse a system, developer or tool message after the first, and a `reasoning_effort` other than
  `low`, `medium` or `high`.
- `llama.config` refuses `mlp_bias` and a Gated DeltaNet convolution of other than 4 taps, and the tokenizer a
  normalizer other than NFC, where they loaded and computed something else.

### Fixed

- A reduction over a leading dimension other than the first, fused with an input of the result's batch, read the
  wrong matrices: `x.sum(1) * y` was off, and so were softmax's and layernorm's gradients over that dimension.
- A matmul under a reduction kept its line of B in L1 when A's row changed along the line, so later passes used the
  wrong row: `(A @ B).softmax(-2)` was off by 0.9.
- `y + x.sum(1)` on 4D, with the sum broadcast along rows or columns, failed to compile.
- A unit-step slice of an inner leading dimension, like `x[:, 0:1]`, had no gradient.
- Casting int32 to a block float gave wrong values: `[1, 2, 3, -4]` came back `[0, 0, 0, nan]`.
- Slicing, relabelling or repeating a block-float repeat, merging a block float, and a block float with a number
  raised; for fp32, the same copies changed -0 and NaN payloads.
- `x / 0` raised `ZeroDivisionError`. It's inf, -inf or NaN as in NumPy, and `number / x` works.
- `gather` and `put` used only the first row of 4D ids of two rows; they refuse them now.
- Through tt-kmd, the chip stayed at its minimum AI clock. `Kmd` asks kmd for the busy one.
- `python3 -m baretile.chip reset` left the card with no driver. It binds tt-kmd again and opens the card through it.
- An ARC message through tt-kmd that timed out made every later message fail.
- A stream whose client left before its headers were written kept its slot, a stream that got no text in 600 seconds
  ended without an error or `[DONE]`, and response ids repeated across requests.
- In the tinygrad adapter, `x ** inf` and `x ** nan` crashed, and so did a `cat` or `stack` of bools from comparisons
  of different dtypes. `BARETILE_REPORT` said every `TinyJit` call replayed when one recorded again or launched
  nothing.

## 0.0.6

The release whose tinygrad programs run on baretile in about baretile's own time. A small GPT-2, 2 blocks 128 wide on 64
tokens in fp32, written once with tinygrad's llm.c model and once with `baretile.models.train`, per call on ttsim:

| | 0.0.5 | 0.0.6 | baretile's own |
| --- | --- | --- | --- |
| tinygrad, a sequence's loss | 228 launches, 2 downloads, 9.78s | 124 launches, 0.92s | 108 launches, 0.94s |
| tinygrad, an AdamW step | 781 launches, 4 downloads, 25.16s | 500 launches, 3.42s | 417 launches, 3.11s |

At 0.0.5 each call downloaded values for the host to compute, so a `TinyJit` ran its Python every call; now both
replay. ttsim's time is the simulator's, not a card's. A small MLP's bf16 inference through tinygrad takes baretile's
own 2 launches.

### Changed

- The tinygrad adapter runs tinygrad's `POW`, like an MSE loss's square, and a `WHERE` on a constant, like its
  gradient's, on the chip, so an SGD step on that loss replays under `TinyJit`.
- tinygrad's bf16 matmul, which sums its bf16 products in fp32 and casts the sum to bf16, is one baretile bf16 matmul,
  and the ops after it, like a bias and an activation, join its kernel. A matmul that more than one kernel reads is
  computed once; one that baretile cuts into parts along K keeps the fp32 sum.
- A matmul is made as `a @ b` or `b @ a`, whichever transposes fewer values, so a gradient like `x.T @ g` needs no
  transpose after it.
- A reduction that moves its axes first keeps the rest as rows and columns, where it made a column a tile row for
  each 32 values, which every op after it was moved into.
- tinygrad's `Embedding` is a `gather`, and its gradient a matmul by the one-hot's transpose, as baretile's own
  `gather`'s gradient is; neither broadcasts the table over the tokens. An id outside the table reads what's there,
  where tinygrad's one-hot gives zeros.
- Attention heads of whole tiles, split from a row of q, k and v, are views of it, a matmul reads one transposed in
  place, and heads side by side are a `merge`.
- tinygrad's `stack`, which its `cat` of equal parts is, runs on the chip, where it went through the host and kept a
  `TinyJit` from replaying. Pads summed, as a `cat` of other parts and a split's gradient are, write each part into
  its columns.
- A `TinyJit` replay writes what the call stores into a captured buffer, like an optimizer's new weight, into the
  buffer's other tensor, and the two take turns, where it copied each back: a copy a parameter a step, three with
  Adam.
- A tensor made from host data on `BARETILE` crosses PCIe once, and isn't copied on the chip after. fp32 that's only
  read as bf16, as `Tensor(a).cast(dtypes.bfloat16)` from NumPy is, is rounded on the host, to nearest even as
  tinygrad's CPU rounds it, and goes up as bf16, with no cast kernel.

### Fixed

- A freed tensor whose address was `Bound`, as `jit` gives a copy of an input it can't read in place, handed the
  `Bound` to the next allocation there, in any runtime on the chip. A later recording that launched with that tensor
  would have bound it to an input, and its replays moved it there.

## 0.0.5

The first release whose tensors have more than three dimensions, and whose Qwen3-0.6B passes the generation check:

```python
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor

rt = Runtime(Chip(pci.open_device()))
x = Tensor.of(rt, [[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]])  # (2, 2, 1, 2)
print(x.sum(1).tolist())
# [[[[4.0, 6.0]]], [[[12.0, 14.0]]]]
```

### Added

- A tensor has up to five dimensions: up to three leading ones, which fold into the batch, then the matrices' rows and
  columns. Elementwise ops and matmuls broadcast over them as NumPy does, reading an operand in place where its leading
  sizes are one run of the result's, and `sum`, `max`, `mean` and `softmax` reduce over any of them in one kernel.
  Movement and slicing take every dimension, a reshape of the leading sizes is a view, and gradients come back through
  all of them. [The public API](docs/api.md#tensors-and-memory) lists the ops that still take three.
- A kernel's programs can `print(fmt, ...)` a string literal and up to six words. The host shows each line after
  `wait()`, with the worker, core and launch that printed it, and a hang's report ends with the last three lines of
  each of the stuck worker's cores. Each core keeps its last 127 lines.
- The tinygrad adapter makes a view a tensor in one move over up to five axes, where a permute or padding on two axes
  of a group took several moves or a copy through the host, so a `TinyJit` replays such calls too.

### Changed

- Qwen3 and Qwen3.5 run the q and k norm in fp32 and round once for RoPE. Qwen3-0.6B now passes the first workload's
  bf16 logit gate, at 2.95% at worst where it was 4.40%, in the same launches.
- A reshape that regroups whole tile rows into other matrices, like `(2, 64, cols)` into `(128, cols)`, is a view.
- Each kernel image holds `kmain` once: 153 of the suite's programs held its body twice.
- Loaded kernels get 16 KiB less room in L1, which the print rings take.

### Fixed

- An operand broadcast into a result whose gradient was broadcast too, as a sum's is, got that gradient once instead
  of once for each copy: `(a + c).sum()` gave `c` a gradient of 1 where it's the number of copies.

## 0.0.4

The first release that can open a card without root, and that trains a model from scratch:

```sh
python3 -m pip install 'baretile[models]'
python3 -m baretile.models.train train.bin val.bin layers=2 dim=64 heads=2 ctx=32 accum=2 steps=10 warmup=2
```

### Added

- On a card that tt-kmd drives, baretile opens it through kmd's device file, `/dev/tenstorrent/N`. That needs no
  root, needs no `iommu=pt` when the IOMMU translates, and leaves the card to tt-smi too. It needs tt-kmd 2.0 or
  later, and 2.11 for an explicit `Chip.arc_msg`.
- `python3 -m baretile.models.train` trains GPT-2 from scratch on GPT-2 tokens in llm.c's `.bin` shards, such as
  FineWeb's. By default that's GPT-2 124M with nanoGPT's recipe: AdamW, a warmup and cosine schedule, and clipping.
  [Training](docs/models.md#training) shows how to get the tokens.
- `BARETILE_DUMP=folder` writes the C programs of each kernel a runtime launches, one file for each core, and
  `baretile.viz` shows them beside the kernel's launches.

### Changed

- With tt-kmd driving the card, baretile no longer unbinds it. It opens the card through sysfs as root, as before,
  only when tt-kmd doesn't drive it, and `python3 -m baretile.chip check` reports what that path needs.
- The Llama families keep the residual stream between blocks in fp32. SmolLM-135M now passes the first workload's
  bf16 logit gate, at 2.72% at worst where it was 4.16%. A decode step takes more launches.
- The core, from the PCIe driver to the tensors, has a size budget: under 4,500 lines of code at 120 columns, with
  every line within 120 columns.

## 0.0.3

The first release that runs models from an installed package:

```sh
python3 -m pip install 'baretile[models]'
python3 -m baretile.chip ttsim
python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000
```

### Added

- The models, the OpenAI-compatible server and the tinygrad adapter are in the
  package, as `baretile.models` and `baretile.adapters`. The `models` extra adds
  NumPy, which they need; baretile itself still needs only the standard library.
- `python3 -m baretile.models.serve` serves a model by its Hugging Face ID.
- `python3 -m baretile.chip ttsim` downloads the tested ttsim release, so
  `TTSIM` is optional.
- `python3 -m baretile.chip check` checks clang, lld, NumPy, and the simulator
  or the card, and says how to fix what's missing.
- `python3 -m baretile.viz -m module` runs a module under the launch viewer.
- The supported model families are listed in
  [docs/models.md](docs/models.md#running-a-model), each compared against
  transformers on ttsim. Other model types are refused at load.

### Changed

- `from models import llama` is now `from baretile.models import llama`, and
  `adapters.tinygrad_baretile` is `baretile.adapters.tinygrad_device`. The
  commands run as `python3 -m baretile.models.serve`, `.llama`, `.gpt2` and
  `.cost`.
- A device has one opener at a time. A second `Chip` on the same chip raises
  `RuntimeError`.
- Closing a runtime lets its running launch finish, for up to two seconds,
  instead of stopping workers inside a Tensix program.
- The server sends a stream's usage chunk only when
  `stream_options.include_usage` asks for it, reads `max_completion_tokens`,
  and takes message content as a list of text parts.
- A jit replay gives each input's gradient back to the caller.

### Fixed

- A matmul split along K dropped its fused scalars, so `(x @ w) * 0.5` was
  wrong for long K.
- fp32 `exp`, `exp2`, `reciprocal` and `where` mishandled NaNs with the sign
  bit set.
- jit replays could read and write the same storage when an input shared it
  with a result or with something the replay writes.
- A runtime stopped in the middle of a Tensix kernel left its workers blocked
  for the next runtime.
- The tokenizer took quadratic time on long words, and its cache grew without
  bound.
- The server could drop a slow client without a 408, kept generating for a
  client that had hung up, failed on padded vocabularies, and lost the text of
  a tool call that was cut short.
- Turning on a card's bus mastering cleared its PCI error bits.
- `baretile.viz` crashed when the script closed or replaced its runtime.

## 0.0.2 and earlier

The device, the runtime, Tensix code generation, the kernels and `Tensor`,
without the models.
