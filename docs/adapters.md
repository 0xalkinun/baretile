# Adapters: tinygrad and PyTorch

[baretile/adapters/tinygrad_device.py](../baretile/adapters/tinygrad_device.py)
lets tinygrad programs, and PyTorch programs through tinygrad's torch frontend,
run on baretile. It's layer 4, beside the models.

## Running a program

Import `baretile.adapters.tinygrad_device` before using tinygrad's `BARETILE`
device, or run an existing tinygrad script through it:

```sh
python3 -m baretile.adapters.tinygrad_device /path/to/script.py
```

The adapter calls tinygrad's internal APIs, so a tinygrad update can break it.
CI tests it on the tinygrad commit pinned in
[.github/workflows/test.yml](../.github/workflows/test.yml). Put a checkout of
that commit on `PYTHONPATH`, or in `TINYGRAD` for the tests.

PyTorch uses tinygrad's `extra/torch_backend`, which needs a tinygrad checkout,
PyTorch, a C++ compiler and ninja. `test_torch_frontend` exercises that path.

## How tinygrad's graph becomes tensors

The adapter translates tinygrad's graph into baretile tensors before tinygrad
schedules kernels. Its tensors have three dimensions, with tinygrad's leading
axes folded into the rows. A view that an operation needs as a tensor stays a
view when it's whole tiles, like the attention heads that GPT-2 splits from a
row of q, k and v, and a matmul reads its transpose in place. Matrices side by
side, as the heads are merged back, are one `merge`. A stack, which is
tinygrad's `cat` of equal parts, is a batch with each part computed into its
matrix, and pads summed, as a `cat` of other parts is, write each part into its
columns. Another view, like a permute or padding, is one move on the chip when
its axes merge into at most five runs, a move for each pair of axis groups it
swaps with more, and anything else is copied through the host.

A broadcast multiply followed by a reduction is a matmul, also when the products
are cast to fp32 before the sum, as in tinygrad's bf16 matmul. The adapter makes
it as `a @ b` or `b @ a`, whichever transposes fewer values, so a gradient like
`x.T @ g` needs no transpose after it. A bf16 matmul's sum cast back to bf16 is
baretile's bf16 matmul, and the elementwise ops after it, like a bias and an
activation, join its kernel. A matmul that more than one kernel reads is
computed once. One that baretile cuts into parts along K keeps the fp32 sum.
Casts between bf16 and fp32 don't end a kernel either
([fusion](compiler.md#fusion-and-reductions)), so a bf16 weight's update by an
fp32 learning rate is one kernel. An fp32 matmul is good to about 2^-11 of its
largest products, far less than fp32 on tinygrad's CPU
([matmul precision](compiler.md#matmul-precision)).

A sum of `where(ids == arange, x, 0)`, as tinygrad's `Embedding` is, is one
`gather` when it sums along the arange, so it never broadcasts the table over
the tokens. An id outside the table reads what's there, where tinygrad's one-hot
gives zeros. Summed along the ids, as the embedding's gradient is, it's a matmul
of the one-hot's transpose.

## Numbers and the CPU

baretile has no fp16 storage, so the adapter keeps tinygrad's fp16 values in
fp32, rounded to fp16 wherever tinygrad makes one (`Tensor.rounded`), to nearest
even. tinygrad's CPU skips the rounding of a half product that its kernel casts
up to fp32, and the adapter doesn't. A cast to bf16 on the chip rounds half up,
where tinygrad's CPU rounds half to even, so the two can differ by one bf16
step.

Unsupported operations fall back to tinygrad's CPU device. These include 64-bit
integers, bit operations, random-number generation, symbolic shapes, `x ** inf`
and `x ** nan`. Int32 multiplication and reductions also use the CPU to keep
exact integer results, since baretile computes them through fp32. A store into
whole rows of a buffer, as a KV cache takes a new position, is one `put` on the
chip; other stores go through the host. The adapter keeps the logical dtype
through fallback, and boolean results stay 0 or 1.

With `BARETILE_REPORT=1`, the adapter prints at exit what it ran on tinygrad's
CPU and why, at the line of your program that ran it, and why a `TinyJit` call
didn't replay. A program that stays on the chip gets one line that says so.

## TinyJit

`TinyJit` uses baretile recordings when the call can be recorded: it records
the second eligible call and replays its launches from the third. A replay reads
each input, and each BARETILE buffer the call captured, where it is at that
call, so new inputs, a weight assigned between calls, and buffers that another
`TinyJit` updates all replay without recording again. What the call stores into
a captured buffer, like an optimizer's new weight, goes into a second tensor for
that buffer, and the replays write the two in turn. Results reuse their recorded
storage.

A call records again when a buffer the recording reads has another layout,
shape or dtype, or when a buffer the recording made or uploaded has changed.
Calls that read data back to the host, bind symbolic variables, pass repeated or
partial input buffers, store into input buffers, store rows of a buffer, or
return an input or a view of one run normally. `BARETILE_REPORT` says which.
