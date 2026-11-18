# baretile documentation

These pages explain how baretile works and how to run it on the simulator or a
card. The [README](../README.md) has the setup and what runs on a card. Each
page links to the files it covers.

## Learning baretile

The ramps teach baretile in order, from tensors down to Tensix instructions.
Each one explains its steps in comments and checks its results, also under
`python -O`, so you can change a step and see what breaks. Run them from the
checkout, as `python3 ramps/1_tensors.py`, after `python3 -m baretile.chip ttsim`
or with `TTSIM` set:

1. [Tensors](../ramps/1_tensors.py): operations, fusion, shapes, formats, views,
   weight storage, gradients and `jit`.
2. [Runtime](../ramps/2_runtime.py): C kernels on the workers' RISC-V cores,
   DRAM and the NoC, the command queue, a tensor's tiles, and timeouts.
3. [Tensix](../ramps/3_tensix.py): unpack, math and pack programs written by
   hand, including a matmul and the SFPU, compared with what `Tensor` compiles.
4. [A kernel](../ramps/4_kernel.py): a tensor operation of your own, which
   [writing a kernel](kernels.md) walks through.

After the ramps, [baretile/models/gpt2.py](../baretile/models/gpt2.py) is a
whole model in under 200 lines, which runs on baretile tensors and NumPy arrays.

## Pages

| Page | What it covers |
| --- | --- |
| [Public API](api.md) | What's stable, what's experimental, and the behavior you can rely on, with the test behind each promise |
| [Runtime](runtime.md) | Workers and the dispatcher, L1 layout, the command ring, DRAM allocation, `jit` and profiling |
| [Compiler](compiler.md) | Tile layout and storage formats, fusion and reductions, unpack, math and pack, matmul and attention |
| [Models and serving](models.md) | The model loaders, precision modes, experts and Gated DeltaNet, the OpenAI-compatible server and cost estimates |
| [Adapters](adapters.md) | tinygrad programs, and PyTorch's through tinygrad, on baretile |
| [Training](training.md) | Training GPT-2 from scratch, and checking it against PyTorch |
| [First P150a workload](workload.md) | Pinned inputs and correctness gates for Llama-3.1-8B-Instruct on one card |
| [Testing and debugging](testing.md) | Running the tests, optional checks, ttsim-chaos and hung launches |
| [Writing a kernel](kernels.md) | A tensor operation of your own, from its four programs to a tested function |
| [Hardware reference](hardware.md) | PCIe, TLB windows, coordinates, DRAM, ARC, the RISC-V cores, NoC transfers and Ethernet, with sources |
| [Bring-up on a card](bring-up.md) | Running on a P150a, giving the card back, resetting it, and what still needs checking |

## How the pieces fit

`bh.py` holds hardware constants, `pci.py` opens the device, and `chip.py` reads
and writes through the network on chip (NoC). The runtime uses these to load and
launch programs. The compiler builds kernels from Tensix instructions, and
`Tensor` compiles expressions into those kernels. `realize()` turns an
expression into kernels and queues them, and `tolist()` also waits and copies
the result back. [CONTRIBUTING](../CONTRIBUTING.md#how-the-code-is-organized)
has the rules for the layers, and [writing a kernel](kernels.md#the-shape-of-a-kernel)
what each RISC-V core of a Tensix tile does in a kernel.

### Terms

A worker is a Tensix tile running a kernel, and BRISC and the TRISCs are its
RISC-V cores. A tensor tile holds 32 by 32 values: the data a worker processes.
"Tile column" refers to `Tensor.columns`.

## Where the code is

| Layer | File | What it does |
| --- | --- | --- |
| 0, device | [bh.py](../baretile/bh.py) | Blackhole's addresses, sizes and instruction encodings, as data |
| | [pci.py](../baretile/pci.py) | The device: `Sim` wraps libttsim.so, `Kmd` opens a card through tt-kmd, and `Card` through sysfs; checking a card first, and giving it back to tt-kmd |
| | [chip.py](../baretile/chip.py) | NoC reads and writes through TLB windows, ARC messages, telemetry, core reset and Ethernet links; `python3 -m baretile.chip [check \| ttsim \| release \| reset]` |
| 1, runtime | [runtime.py](../baretile/runtime.py) | Builds device C with clang, loads firmware and kernels, queues launches, allocates DRAM and profiles |
| 2, codegen | [tensix.py](../baretile/tensix.py) | Tensix instructions from the ISA, and the BRISC, unpack, math and pack programs of a kernel |
| 3, kernels and tensors | [kernels.py](../baretile/kernels.py) | Fused elementwise, matmul and reduction kernels, attention, data movement and `topk` |
| | [tensor.py](../baretile/tensor.py) | `Tensor`: lazy expressions, broadcasting, views, storage formats, autograd and `jit` |
| Commands | [hello.py](../baretile/hello.py), [bench.py](../baretile/bench.py), [probe.py](../baretile/probe.py), [viz.py](../baretile/viz.py) | The fib demo, a matmul benchmark, card measurements and the launch viewer |
| 4, on top | [baretile/models/](../baretile/models) | GPT-2 and its training, the Llama loader and its families, Gated DeltaNet and experts (`qwen.py`), the server, the cost model and a decode step's benchmark |
| | [baretile/adapters/](../baretile/adapters) | The BARETILE device for tinygrad |
| | [ramps/](../ramps), [test/](../test) | The ramps, and the tests with the bring-up and workload scripts |
