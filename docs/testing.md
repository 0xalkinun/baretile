# Testing and debugging

Every part of baretile has a test that runs on ttsim. This page covers running
the tests, the optional checks that need models or other packages, how to read
the programs a kernel runs, and what to do when a kernel hangs.

## Running the tests

Set up baretile and ttsim as [Try it](../README.md#try-it) does, then run these
from the checkout, with clang and ld.lld on `PATH`:

```sh
python3 -m pip install numpy pytest ruff
python3 -m baretile.chip ttsim
python3 -m pytest -s test
ruff check
```

The tests use the simulator that `python3 -m baretile.chip ttsim` downloads, or
the `libttsim.so` that `TTSIM` names. The whole suite takes several minutes on
ttsim. With [pytest-xdist](https://pypi.org/project/pytest-xdist/) installed,
`-n auto` runs the tests in a process for each CPU, each with its own
simulator, as CI does.

The tests follow the layers of the code. `test_device.py` starts at PCIe access
and goes through the runtime. `test_kernels.py` tests the instructions and the
tensor operations built from them. Models, the server, the adapters and the
[ramps](../ramps) are in `test_models.py`. So are the doc checks:
`test_readme_examples` runs the README's Python examples and checks what they
print, `test_doc_links` checks every link to a file of the repository and its
anchor, and `test_website_matches_readme` checks the homepage's code against the
README's. `test_layers` checks that each module imports only the layers below
it, and `test_size` the core's line budget. Each test seeds `random` from its
own id, so a failure repeats in any order or worker.

If pytest exits without explaining why, run it with `-s` and without `-n`. ttsim
stops the process on errors such as an invalid address or decrementing an empty
semaphore, and pytest can otherwise hide that error message. The tests share a
`Chip` because the simulator can only be initialized once per process.

The runtime compiles device programs with clang for `riscv32-unknown-elf` and
links them with lld. Builds go in `$XDG_CACHE_HOME/baretile`, which is
`~/.cache/baretile` by default, so you won't have to compile them again on the
next run. Entries expire after a week without use.

`PART=i/n` runs part i of n of the tests, split so the parts take about as long
by the seconds each test took in [test/durations.json](../test/durations.json).
`DURATIONS=write` rewrites that file from the tests a run runs, so run the whole
suite with it after adding slow tests. `SKIPS=fail` turns a skip into a
failure.

## Optional checks

Most model tests use small configurations. To compare against transformers,
install PyTorch and a transformers release that includes the model you're
testing. The MXFP4 test also needs safetensors, `test_tokenizer` needs
tiktoken, and the adapter tests need tinygrad at the commit that CI pins (see
[adapters](adapters.md#running-a-program)). With a checkout in `TINYGRAD`, they
fail instead of skipping when tinygrad doesn't import. Check pytest's skipped
tests if you expected one of these comparisons to run, or set `SKIPS=fail`.

CI's test job doesn't install these, so its `references` job does, at pinned
versions, and runs the comparisons with `SKIPS=fail`: `test_train`,
`test_train_gpt2`, `test_train_reference`, `test_llama_frequencies`,
`test_tokenizer`, `test_tokenizer_gate`, `test_hf_families`, `test_qwen3_moe`,
`test_qwen3_5_moe`, `test_mxfp4`, `test_torch_frontend`,
`test_autonomous_generation`, `test_workload_commands` and
`test_workload_blackhole_py`. It downloads no checkpoints.

To include pretrained checkpoint checks:

```sh
GPT2=/path/to/gpt2/model.safetensors python3 -m pytest -s test -k gpt2_model
LLAMA=/path/to/SmolLM-135M python3 -m pytest -s test -k llama_model
```

Those follow NumPy's choices. The [first workload](workload.md#correctness-gates)
has the commands that hold a model to its gates: generating from its own picks,
and agreeing with Tenstorrent's reference tokens.

`LLAMA8B=1` enables a test at Llama-3.1-8B block dimensions. `FUZZ=n` runs n
random expressions in the kernel fuzz test instead of 30, when n is more.
`TTSIM2=/path/to/libttsim.so` enables the two-chip test. ttsim's releases
include that library as `libttsim_bh_x2.so`, or you can build ttsim's
`release_bh_x2` target. `KMD_IOCTL=/path/to/ioctl.h` checks the arguments of
the ioctls that `pci.Kmd` sends against tt-kmd's header, compiled by clang for
this host. CI uses the header from tt-kmd 2.11.0.

## Benchmarks

[test/benchmark.py](../test/benchmark.py) runs matmul, decode, prefill and
serving benchmarks on baretile, and the first three on TT-NN on the same card,
and compares the two. Its opening comment gives the commands and workload
sizes. The TT-NN runs need tt-metal's Python environment, and through sysfs the
card [given back](bring-up.md#giving-the-card-back) first. It also needs
PyTorch to read the reference file, and `cloc` to count lines of code. `load`
drives any OpenAI-compatible completion server.

## Where ttsim differs from a card

ttsim models a p150, the Blackhole board that the P150a is a version of. Tensix
instructions take no time in ttsim. NoC requests complete immediately.
`STALLWAIT` and `SEMWAIT` hold up the whole thread, whatever their block masks
name, so a wait that names too few units still works there. Most of the cycles
you see come from the RISC-V code that controls the kernel. A lower count can
show that you've reduced that work, but it won't tell you how fast the kernel is
on a card.

ttsim also decodes an unexpected `XMOV` in very long loops of inline Tensix
instructions. We don't know why. The DeltaNet kernel calls SFPU functions to
avoid generating a loop that long.

The [hardware reference](hardware.md) notes what a card does that ttsim
doesn't, and [bring-up](bring-up.md#what-still-needs-checking) lists what we
can only check on a card.

## ttsim-chaos

[test/ttsim-chaos.patch](../test/ttsim-chaos.patch) adds NoC delays, FPU and
SFPU latency, random pauses, a 140-tile option, the NIU counter for read bytes
and a tile's NoC counters left nonzero to ttsim v1.11.5 at commit `464814f`.
Apply it to a separate ttsim checkout and build the Blackhole library there:

```sh
git clone --branch v1.11.5 https://github.com/tenstorrent/ttsim ttsim-chaos
cd ttsim-chaos
git apply /path/to/baretile/test/ttsim-chaos.patch
./make.py src/_out/release_bh/libttsim.so
```

Run with delays enabled to check whether the kernels wait for their data:

```sh
TTSIM=/path/to/ttsim-chaos/src/_out/release_bh/libttsim.so \
  TTSIM_CHAOS_NOC=1,400 TTSIM_CHAOS_RISCV=0.2 TTSIM_CHAOS_TENSIX=0.2 TTSIM_CHAOS_MATH=1,40 \
  python3 -m pytest -s test
```

`TTSIM_CHAOS_NOC=lo,hi` delays a request by a random number of clocks in that
range. The RISC-V and Tensix settings give the probability of skipping a turn.
Set `TTSIM_CHAOS_SEED` to repeat a run. `TTSIM_CHAOS_UNHARVESTED=1` enables all
140 Tensix tiles, so you can check the code with a larger grid.

`TTSIM_CHAOS_MATH=lo,hi` gives the FPU and SFPU time. Each of their instructions
runs a random lo to hi clocks after it leaves its thread, and the thread goes on
meanwhile. `STALLWAIT` and `SEMWAIT` then block only the instructions their
block masks name, as the ISA's `STALLWAIT.md` describes. A configuration write,
a `SEMPOST` or a pack that should wait for the math finds it unfinished unless a
`STALLWAIT` holds it back. FPU and SFPU instructions keep their order, and each
one uses the address modes it saw when it left its thread, since tt-llk relies
on both. The unpackers, the packer and the configuration unit still finish at
once, so this doesn't check the waits for them.

`TTSIM_CHAOS_OUTSTANDING=x,y` starts Tensix tile (x, y) with 200 requests
counted as outstanding on NoC 0 for transaction ID 0, and `x,y,n` with n, up to
255, as a P150a's firmware leaves a tile after a reset
([#4](https://github.com/0xalkinun/baretile/issues/4)). A new runtime zeroes the
counters before its firmware starts, so with `TTSIM_CHAOS_OUTSTANDING=1,2`,
`python3 -m baretile.hello` passes. Without that, it would hang at launch 1 on
(1, 2), with `NoC requests in flight by id: {0: 200}` in the report.

The patch also implements the NIU counter for bytes received by reads.
`pci.Sim.counts` detects the patched library. Released ttsim doesn't support
that counter, so profiles from it report `None` for bytes.

For every push to main and every pull request that changes more than Markdown,
CI runs the device and kernel tests, the DeltaNet and MoE kernel checks
(`test_delta_step`, `test_delta_pieces` and `test_moe`) and the kernel ramp
under ttsim-chaos, with the four delay settings above, the run number as the
seed and `TTSIM_CHAOS_OUTSTANDING=1,2`. It runs them as parts, and a process's
noise follows from the seed and the tests it runs, so
`TTSIM_CHAOS_SEED=s PART=i/n` with the job's settings repeats a failed part. CI
also runs the whole suite with `TTSIM_CHAOS_UNHARVESTED=1`.

## Reading a kernel's programs

A kernel is up to four C programs, one each for BRISC and the three TRISCs,
which Python writes for each kernel it builds. To read the programs a script
ran, set `BARETILE_DUMP` to a folder:

```sh
BARETILE_DUMP=kernels python3 ramps/1_tensors.py
```

Each kernel a runtime launches is written there once, in a folder named after
the kernel and a hash of its image, such as `matmul-relu-1a2b3c4d/`. It has a
`.c` file for each core with a program. Every file starts by including
`device.h`, which holds the helpers that baretile puts before each program, so
with `device.h` pasted in its place, a file is exactly what clang compiled.
Diffing two dumps shows what a change did to the generated code. The
[launch viewer](runtime.md#profiling) shows the same programs for the kernel you
select.

## Printing from a kernel

A kernel's programs can call `print` with a format and up to six words, as they
would call `printf`:

```c
print("tile %u of %u: %#x\n", t, a[1], value);
```

`Runtime.wait()` shows each line once the launch is done, after the worker,
the core and the launch that printed it:

```
(1, 2) BRISC, launch 3 (add_relu): tile 0 of 4: 0x3f80
```

The core leaves a record in a ring in its L1, and the host does the formatting,
so a call is a few dozen instructions. The format must be a string literal,
which the host reads from the kernel's image. Each argument is a 32-bit word:
`%d` reads it as a signed int, `%u` and `%x` as an unsigned one, and `%f` as the
bits of an fp32, so pass a float `x` as `*(unsigned *)&x`. There's no `%s`, and
a line whose format doesn't match its words is shown raw.

Lines come grouped by worker, then by core. Each core's lines are in the order
it printed them, but its four cores run at once, so lines from different cores
aren't in time order. Each core's ring keeps its last 127 lines, and a line says
how many came before the lines shown. A kernel that doesn't call `print`
compiles to the same bytes. [test_print](../test/test_device.py) shows each
case.

## When a launch hangs

`Runtime.wait()` times out after 60 seconds without a completed launch by
default. Its report identifies the current launch and unfinished workers. For
up to four workers, it prints core status, synchronization counters, flags and
outstanding NoC requests. On a card it also reads the program counters. If the
kernel [prints](#printing-from-a-kernel), the report ends with the last three
lines that each of the worker's cores printed, which can show where it stopped.

The 60 seconds are wall time, on ttsim too, where a launch takes as long as the
simulator needs to run it. A large launch, or a machine that's busy with other
simulators, can make a launch take longer without hanging. A real hang shows the
same counters in reports a minute apart. For long checks, run one simulator at
a time, or pass a longer timeout to `wait()`.

If a worker never received `go`, check dispatch and multicast first. If the
workers finished but the host is still waiting, check their atomic reports and
the status write to host memory. A read ID that's still outstanding means a
transfer hasn't finished. Use the
[pipeline counters](compiler.md#unpack-math-and-pack) to see where a kernel is
waiting.

Creating a new runtime resets its worker cores, but not a Tensix coprocessor
that a hung kernel left waiting. If the next runtime's kernels hang on those
workers, or the NoC or ARC has hung, see
[bring-up](bring-up.md#when-a-card-hangs) for the chip reset command.
