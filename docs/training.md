# Training

`python3 -m baretile.models.train` trains GPT-2 from scratch, and tinygrad and
PyTorch programs train through the [adapter](adapters.md). The tests check
both against PyTorch. Nothing larger than the tests' models has trained on a
card yet ([#9](https://github.com/0xalkinun/baretile/issues/9)).

## GPT-2 from scratch

[baretile/models/train.py](../baretile/models/train.py) trains on GPT-2
tokens. Its defaults are GPT-2 124M's shape and nanoGPT's recipe:

- 12 blocks 768 wide, with 12 heads, on sequences of 1,024 tokens, and the
  vocabulary padded to 50,304.
- Weights, gradients and AdamW's state in fp32. AdamW uses betas 0.9 and 0.95,
  and decays the matrices and embeddings by 0.1, but not the biases or the
  layer norms. Gradients are clipped to a norm of 1.
- Matmuls in fp32, each three bf16 matmuls (see
  [matmul precision](compiler.md#matmul-precision)). `precision=bf16`
  multiplies bf16 operands instead, as PyTorch's autocast does, and keeps
  everything else in fp32.
- 524,288 tokens a step, as the gradients of 512 sequences added up. The
  learning rate warms up to 6e-4 over 700 steps, then follows a cosine down to
  a tenth of that at the last of 19,073 steps, which read 10B tokens once.

Each sequence's forward and backward pass, and each update, runs as a recorded
replay (`jit`), so the host only sends the sequence's tokens. The chip compares
the targets with each column of the logits to pick them out, so no one-hot
matrix crosses PCIe.

The tokens come from llm.c's `.bin` shards: a 1 KiB header whose first three
int32s are 20240520, 1 and the token count, then the tokens as uint16s.
FineWeb's 10B-token sample comes in that form. The run reads its training shards
in the order given, then the first again, and with two or more files the last is
the validation shard. #9's run reads ten shards, 1B tokens:

```sh
hf download kjj0/fineweb10B-gpt2 --repo-type dataset --local-dir fineweb fineweb_val_000000.bin $(seq -f fineweb_train_%06g.bin 1 10)
python3 -m baretile.models.train fineweb/fineweb_train_*.bin fineweb/fineweb_val_000000.bin precision=bf16 steps=1907 log=train.json save=train.npz
```

Settings go after the files as `key=value`; run the command with no arguments
to list them. On ttsim, try a small model, such as
`layers=2 dim=64 heads=2 ctx=32 accum=2 steps=10 warmup=2`. Every 250 steps
(`val_every`) the run reports the validation loss over the same 20 sequences of
the validation shard, and `log` writes each step's numbers as a line of JSON.

With `save`, the run keeps its weights, AdamW's state and its step in that file
every 100 steps (`save_every`) and at the end, and replaces the file only once
the new one is on disk. The same command run again resumes after the saved
step. It refuses a file saved with other settings or shards, though `log`,
`save`, `save_every`, `val_every` and `val_batches` may change. The steps after
the last save run again, so the log can hold such a step twice, with the same
loss.

## Checking against PyTorch

- `test_train_gpt2` trains a small GPT-2 this way on a counting sequence, in
  each precision. At each step, the loss and gradients must match PyTorch's on
  the same weights and sequences, under autocast for bf16, and the update must
  be AdamW's, computed in float64 from the gradients the chip added up.
- `test_train_resume` checks that a run stopped and resumed ends as one that
  ran straight through, bit for bit.
- [test/train_reference.py](../test/train_reference.py) runs the same recipe in
  PyTorch, from the same initial weights and on the same sequences, and writes
  its log in the same form. It runs on CUDA if PyTorch has it, and `stop` ends
  it after that many steps of the full schedule. `test_train_reference` checks
  it against baretile for a small GPT-2 in fp32.

```sh
python3 test/train_reference.py fineweb/fineweb_train_*.bin fineweb/fineweb_val_000000.bin pytorch.json precision=bf16 steps=1907 stop=100
python3 test/train_reference.py compare train.json pytorch.json
```

[test/train.py](../test/train.py) trains a small character-level GPT-2 written
in tinygrad, llm.c's GPT from tinygrad's examples with tinygrad's Adam, on the
README and these docs, through the adapter. At each step PyTorch computes the
loss and gradients on baretile's current weights, and a separate PyTorch run
trains its own copy from the same starting weights. It needs PyTorch and a
tinygrad checkout:

```sh
python3 test/train.py 300
```

To train your own model, write it with `Tensor`, as `baretile.models.train`
does: its operations have gradients (`backward`, `gradient`), `assign` updates a
weight in place, and `jit` records a whole step. tinygrad and PyTorch programs
train through the adapter, as `test_tinygrad` and `test_torch_frontend` do.

## Estimating a run

`python3 -m baretile.models.cost train` estimates a step and a run of
`baretile.models.train`, at its defaults or with its settings, such as
`precision=bf16` or `steps=1907`, and a card's, such as `probe=probe.json`:

```text
GPT-2, 12 blocks 768 wide, 124.5M parameters: 524,288 tokens a step, 19,073 steps, 10.0B tokens
  precision   TFLOP a step   matmuls, s   GB read   reads, s   step, s   tokens/s   all steps, hours
  fp32               488.6         8.66    24,647      68.77     68.77      7,624              364.3
  bf16               488.6         3.22    10,902      30.42     30.42     17,236              161.2
```

These are estimates. The model counts the matmuls' FLOPs and fidelity phases:
each matmul's backward is two more, and an fp32 matmul is three bf16 ones. The
bytes a sequence reads come from ttsim-chaos's read counter, fitted as
`TRAIN_READS` in [cost.py](../baretile/models/cost.py) describes, which counts
every NoC read and no writes. So they're counts on the simulator, not DRAM
traffic measured on a card, and the times assume 70% of 512 GB/s. Launches, the
SFPU's work and RISC-V control are left out.
