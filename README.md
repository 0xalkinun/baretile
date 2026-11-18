# baretile

[![test](https://github.com/0xalkinun/baretile/actions/workflows/test.yml/badge.svg)](https://github.com/0xalkinun/baretile/actions/workflows/test.yml)

baretile is a compiler and runtime for Tenstorrent Blackhole, written in Python.
You write tensor operations, and it compiles them into Tensix instructions. It
also handles the PCIe access needed to load and run those programs on the chip.

The code in `baretile/` uses only the Python standard library. The driver runs
in userspace, and baretile loads its own firmware onto the workers. It needs
Tenstorrent's board firmware for ARC and Ethernet. There's no dependency on
tt-metal, tt-umd or LLK.

The core, from the PCIe driver to the tensors, is small enough for one person to
read: under 4,684 lines of code at 120 columns, not counting comments and
docstrings. `test_size` keeps it there, and the models, the server and the
commands are on top of it.

## Status

baretile is pre-alpha. The whole test suite runs on Tenstorrent's ttsim
simulator for every change, and CI's `card` job runs
[bring-up](https://github.com/0xalkinun/baretile/blob/main/docs/bring-up.md) on
a P150a for every push to main: all of `test_device.py`, `test_kernels.py` and
`test_models.py`, then SmolLM-135M's checkpoint against NumPy. The tests that
need PyTorch, transformers, tinygrad or a process of their own skip there.

Not in that job yet: the probe, which measures the card, and the
[first workload](https://github.com/0xalkinun/baretile/blob/main/docs/workload.md)'s
Llama-3.1-8B gates ([#5](https://github.com/0xalkinun/baretile/issues/5)).
Nothing larger than the tests' models has trained on a card
([#9](https://github.com/0xalkinun/baretile/issues/9)), and
[bring-up](https://github.com/0xalkinun/baretile/blob/main/docs/bring-up.md#what-still-needs-checking)
lists what else still needs checking there.

## Try it

You'll need Linux, Python 3.12 or later, and clang with a RISC-V target and lld.
On Debian and Ubuntu,
`sudo apt update && sudo apt install python3-venv clang lld curl git` installs
them, with curl and git for the later sections. Install baretile in a virtual
environment, download the simulator and check the setup:

```sh
python3 -m venv baretile-env
. baretile-env/bin/activate
python3 -m pip install baretile
python3 -m baretile.chip ttsim
python3 -m baretile.chip check
```

In fish, activate the environment with `. baretile-env/bin/activate.fish`.

`chip ttsim` downloads [ttsim](https://github.com/tenstorrent/ttsim) v1.11.5,
the release baretile is tested with, which baretile uses when there's no card on
the PCI bus. `chip check` says what's missing and how to fix it. Then `chip`
prints the firmware, the clock, the Tensix tiles and the DRAM banks, and `hello`
runs a program on every worker:

```sh
python3 -m baretile.chip
python3 -m baretile.hello
```

On ttsim, which has 120 Tensix tiles, `baretile.hello` prints:

```text
119 Tensix workers computed fib(30) = {832040}, and one tile dispatched the work
```

To use another ttsim build, such as
[ttsim-chaos](https://github.com/0xalkinun/baretile/blob/main/docs/testing.md#ttsim-chaos),
set `TTSIM=/absolute/path/to/libttsim.so`, which comes before a card and the
downloaded release. Without `TTSIM`, baretile opens a Blackhole through tt-kmd,
Tenstorrent's kernel driver, or as root through sysfs; see
[bring-up](https://github.com/0xalkinun/baretile/blob/main/docs/bring-up.md)
first.

## Tensors

Save this as `tensors.py` and run it with `python3 tensors.py`:

```python
from baretile import pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor

rt = Runtime(Chip(pci.open_device()))
x = Tensor.of(rt, [[1.0, 2.0], [3.0, 4.0]])
w = Tensor.of(rt, [[2.0, 0.0], [0.0, 3.0]])
print((x @ w).relu().tolist())
# [[2.0, 6.0], [6.0, 12.0]]
```

The matmul and ReLU run when you call `tolist()`, in one kernel: the ReLU works
on the matmul's result in the math unit's registers, before it's written out.
Tensors have up to five dimensions and support broadcasting, reductions, views
and autograd, with bf16, fp32, int32, bfp8 and bfp4 storage. To see the kernels
a script launched, run it with `python3 -m baretile.viz tensors.py` and open the
`viz.html` it writes.

## Run a model

The models and the server also need NumPy. Install them, then serve a model by
its Hugging Face ID:

```sh
python3 -m pip install 'baretile[models]'
python3 -m baretile.models.serve HuggingFaceTB/SmolLM-135M 8000
```

The first time, the server downloads the model, about 540 MB. When it prints
`serving SmolLM-135M on port 8000`, ask it for a completion from another
terminal:

```sh
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The meaning of life is", "max_tokens": 4, "temperature": 0}'
```

On ttsim the answer's text is `" a question that has"`. It takes about two
minutes, since the simulator runs every core of the chip, and curl prints
nothing until it's done. The server speaks OpenAI's completions and chat
completions API.
[Models](https://github.com/0xalkinun/baretile/blob/main/docs/models.md) lists
the families baretile runs and the server's options.
[Training](https://github.com/0xalkinun/baretile/blob/main/docs/training.md)
trains GPT-2, and
[adapters](https://github.com/0xalkinun/baretile/blob/main/docs/adapters.md) run
tinygrad and PyTorch programs on baretile.

## Learn baretile

The ramps are scripts that teach baretile from the top down, in the repository.
Start with the first:

```sh
git clone https://github.com/0xalkinun/baretile
cd baretile
python3 ramps/1_tensors.py
```

The [documentation](https://github.com/0xalkinun/baretile/blob/main/docs/README.md)
lists the ramps and explains the runtime, the compiler, the models and the
hardware, with a map of the code.
[CONTRIBUTING.md](https://github.com/0xalkinun/baretile/blob/main/CONTRIBUTING.md)
says how to run the tests, what a change needs and how to report a bug.

Licensed under
[Apache-2.0](https://github.com/0xalkinun/baretile/blob/main/LICENSE).
