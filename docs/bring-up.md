# Bring-up on a card

These steps take baretile onto a Blackhole P150a. Start at the bottom of the
stack and stop at the first failure, since the later tests need the earlier
ones to work. [The README](../README.md#status) says which stages CI runs on a
card.

## Before you start

Check that the card appears in `lspci -d 1e52:b140`, has the required power,
and has working board firmware. Install tt-kmd 2.5 or later, Tenstorrent's
kernel driver, and check the firmware with Tenstorrent's tools. Then check what
baretile needs:

```sh
python3 -m baretile.chip check
```

It reads sysfs without touching the card, so it runs without root. It checks
clang, lld and NumPy, then how baretile will open the card, whether the PCIe
link runs at the card's best speed and width, and the firmware version. It
exits with an error if anything would stop baretile.

With tt-kmd driving the card, baretile opens its device file,
`/dev/tenstorrent/0`, as `pci.Kmd`, which needs no root. The check fails a
tt-kmd older than 2.5, and checks that you can open the file. kmd pins the
memory that the card reaches by DMA. With an IOMMU translating, that's ordinary
memory; otherwise it's a hugepage, so the check also counts the free ones.
tt-smi can use the card at the same time.

Without tt-kmd, baretile opens the card through sysfs as `pci.Card`, which
unbinds any other driver. The check reports whether you're root, whether the
IOMMU passes DMA addresses through, and how many hugepages are free. Configure
the IOMMU for passthrough (`iommu=pt`) or disable it, and reserve hugepages as
root with `echo 4 > /proc/sys/vm/nr_hugepages`.

## Steps

Run bring-up from the checkout, with `TTSIM` unset, in the Python environment
that has the test dependencies. Through sysfs, run it as root:

```sh
python3 test/bringup.py record
```

It runs the stages below in order and stops at the first failure, saying where
the failing stage's output is and what to look at. Each stage's output goes in
`record/<stage>.log`, and `record/record.json` holds the device, the host, the
commit, and each stage's result and time, so keep the folder as the record of
the bring-up. After a fix, `--from <stage>` starts again at that stage, and
`--only` runs the stages you name, separated by commas. An unknown stage name is
an error, and a stage that `--only` names fails when its files aren't given.

1. **preflight** runs `python3 -m baretile.chip check`, as above.
2. **chip** runs `python3 -m baretile.chip`: ARC status, the clock, available
   columns and DRAM training, before any worker program loads.
3. **transfers** runs `test_config_space`, `test_arc`, `test_dma`, `test_l1` and
   `test_dram` from `test/test_device.py`: register access and memory transfers
   that the host drives. Nothing runs on the workers yet.
4. **launches** runs `python3 -m baretile.hello`, then all of
   `test/test_device.py`, including `test_host_memory`, which checks that a
   worker can read and write host memory.
5. **math** runs `test_tile_math` and `test_tensor_math` from
   `test/test_kernels.py`: instruction encoding and basic unpack, math and pack.
6. **probe** runs `python3 -m baretile.probe 8 record/probe.json`, which
   measures launch overhead, transfers, memory latency and Tensix rates, and
   records the AI clock it measured at. Compare `upload over PCIe` with
   `upload over PCIe, through the BAR`: `tensor.DMA_UPLOAD` picks between the
   two by size, and its value is untuned.
7. **kernels** runs all of `test/test_kernels.py`. If a failure comes and goes,
   check the waits that ttsim doesn't model, which
   [ttsim-chaos](testing.md#ttsim-chaos) models in part.
8. **models** runs `test/test_models.py`.
9. **smollm** runs `test_llama_model` on SmolLM-135M's checkpoint, given by
   `--smollm path/to/SmolLM-135M`.
10. **llama**, **timing** and **http** run the [first workload](workload.md)'s
    reference agreement, timing and HTTP gates, given
    `--llama path/to/Llama-3.1-8B-Instruct` and
    `--reference path/to/Llama-3.1-8B-Instruct.refpt`, and keep the reports in
    `record/`.

The stages run on ttsim too, which is how they're tested. After the probe,
`probe=record/probe.json` gives `python3 -m baretile.models.cost` the card's
rates. `python3 -m baretile.bench` and `python3 -m baretile.models.bench`
time a matmul, and a decode or verify step at Qwen3.5-9B's shapes, each against
what the card could do.

## Running CI on the card

The `card` job in [the workflow](../.github/workflows/test.yml) runs
`test/bringup.py` from preflight through the smollm stage, leaving out the probe
and the Llama stages, and keeps the record as an artifact. It waits for the card
while another process has it open. It runs once the repository variable `CARD`
is `true`, and the smollm stage when `SMOLLM` names a SmolLM-135M folder on the
card's machine. It isn't a required check yet
([#11](https://github.com/0xalkinun/baretile/issues/11)).

A self-hosted runner runs whatever a pull request's code does, with its user's
access. So the job never runs for a pull request from a fork, and the runner, a
GitHub Actions runner service labeled `p150a`, runs as its own user,
`baretile-ci`, which can reach the card, the checkout and its own files, and
nothing in other users' homes.

## Giving the card back

Through tt-kmd, baretile never takes the card away, `chip reset` included, and
TT-NN can use it once no baretile process has it open. Through sysfs, `pci.Card`
unbound tt-kmd, so once no baretile process is running, give the card back to it
so that tt-smi and TT-NN can use it:

```sh
python3 -m baretile.chip release
```

## When a card hangs

A new runtime resets the cores after a kernel hangs. If ARC or the NoC stops
responding, close processes using the card before trying
`python3 -m baretile.chip reset`, as root. That writes PCIe configuration, then
removes and rescans the device, and gives the card 10 seconds to come back. With
tt-kmd loaded, the rescan binds the card to it again, and the command opens the
card through it. We've tested the reset against fake sysfs files only, so we
don't know yet whether it recovers a hung card, or whether 10 seconds are
enough. Tenstorrent's `tt-smi -r 0` has brought a P150a back after it left its
PCIe link.

## What still needs checking

- `chip reset` on a hung card, [as above](#when-a-card-hangs), and
  `chip release`, which is tested against fake sysfs files only.
- `pci.Card`'s path through sysfs, whose DMA to host memory takes the
  [direct path](hardware.md#host-memory-and-arc): the card's machine uses
  tt-kmd.
- The BAR2 offsets and parts of ARC's protocol come from the drivers and
  simulator because the ISA doesn't document them.
- The SFPU programs read -1.0 from LReg 11, which leaving the SFPU's soft reset
  sets and tt-llk keeps there. baretile doesn't reset the SFPU itself; the
  card's SFPU tests pass, but nothing reads the register itself.
- The host relies on x86 ordering between writes to the ring and the doorbell.
  There's no explicit barrier for other host architectures.
- ISA and simulator differences: `UNPACR` fields have different names,
  `SETDVALID` is unavailable in ttsim, and some address-mode documentation
  describes Wormhole. The code uses the encodings accepted by the Blackhole
  simulator.
