# Time what a decode or verify step runs at Qwen3.5-9B's shapes, each against what the card could do: the weights' and
# the cache's DRAM reads against the probe's rate (or the GDDR6's 512 GB/s without a probe record), a prompt's matmuls
# against the Tensix peak. Times are the chip's, from each launch's go to its last report. Write the rows as JSON if
# given a file. Speed changes carry its table before and after.
#   python3 -m baretile.models.bench [probe.json] [rows.json]
import json, sys
from types import SimpleNamespace
import numpy as np
from baretile import bh, pci
from baretile.chip import Chip
from baretile.runtime import Runtime, compile_kernel
from baretile.tensix import tile_bytes
from baretile.tensor import Tensor
from baretile.models import llama, qwen, serve

# Qwen3.5-9B: hidden 4096, MLP 12288 (gate and up as one matmul), 16 query heads and 4 key heads of 256, Gated
# DeltaNet with 16 key heads and 32 value heads of 128, and 248,320 rows in the output layer. Its weights are bfp4 at
# one fidelity phase, its KV cache bfp8 and attention at four phases.
DIM, MLP, HEADS, KV, HEAD, KH, VH, HD, VOCAB = 4096, 12288, 16, 4, 256, 16, 32, 128, 248320
MATMULS = (("MLP gate and up", DIM, 2 * MLP), ("MLP down", MLP, DIM), ("attention q and gate", DIM, 2 * HEADS * HEAD),
           ("a square projection", DIM, DIM))
DRAM_PROBE = "DRAM reads of bfp4 chunks, 4 in flight: grid workers, own banks"  # what the probe calls the weights' rate

def table(chip, probe=None):
  """Rows of (what, microseconds, rate, unit, % of the ceiling or None), timed on the chip from each launch's go to
  its last report, so the host's time doesn't count. The values are zeros or random: no operation's time depends on
  them, except the top-k's, which gets random logits."""
  rt, clock = Runtime(chip), chip.aiclk * 1e6
  dram = (probe or {}).get(DRAM_PROBE, bh.DRAM_GBPS)
  peak = len(rt.grid[0]) * len(rt.grid[1]) * bh.MVMUL_FLOPS * clock / 1e12  # TFLOPS at one fidelity phase
  def seconds(make, repeat=5):  # each call's chip time, after a call that compiles and loads
    make(); rt.wait()
    since = rt.launches
    for _ in range(repeat): make()
    return sum(cycles for _, _, cycles, _ in rt.profile(since)) / repeat / clock
  rows = []
  def row(what, s, amount=None, unit="", ceiling=None):
    rate = amount / s / 1e9 if amount is not None else None
    rows.append((what, s * 1e6, rate, unit, rate / ceiling * 100 if rate is not None and ceiling else None))
  empty = compile_kernel("void kmain(volatile unsigned *a) {}")
  row(f"an empty launch on {len(rt.cores)} workers", seconds(lambda: rt.launch(empty, (), len(rt.cores))))
  zeros = lambda r, c, b=1, d="bf16": Tensor.zeros(rt, r, c, b, d).realize()
  weights = {}
  for what, k, n in MATMULS + (("the output layer", DIM, VOCAB),):
    w = weights[what] = zeros(k, n, 1, "bfp4").columns().realize()
    bytes_ = -(-k // 32) * -(-n // 32) * tile_bytes("bfp4")
    for m in (32, 64, 128, 256) if what != "the output layer" else (32,):
      x = zeros(m, k)
      s = seconds(lambda: x.matmul(w, fidelity=1).realize())
      row(f"{what}, {k} x {n} bfp4, {m} rows", s, bytes_, "GB/s of weights", dram)
      if m == 256: row(f"{what}, {k} x {n} bfp4, {m} rows: its math", s, 2 * m * k * n * 1e-3, "TFLOPS", peak)
    del weights[what]
  for keys in (4096, 32768):  # attention over the cache: a verify step's 32 rows of every query head
    q, places = zeros(32, HEAD, HEADS), Tensor.of(rt, [[keys - 1]] * 32, "int32")
    k, v = (zeros(keys, HEAD, KV, "bfp8").repeat_interleave(HEADS // KV) for _ in range(2))
    cache = 2 * KV * -(-keys // 32) * (HEAD // 32) * tile_bytes("bfp8")
    row(f"attention, 32 rows over {keys} cached positions", seconds(lambda: q.attend(k, v, places, fidelity=4).realize()),
        cache, "GB/s of the cache", dram)
  # a draft pass's 6 rows over the drafter's 8k bf16 window, each key head's 4 query heads as one tile row, as llama
  # groups a few rows
  keys, q, places = 8192, zeros(6 * HEADS // KV, HEAD, KV), Tensor.of(rt, [[8191]] * (6 * HEADS // KV), "int32")
  k, v = zeros(keys, HEAD, KV), zeros(keys, HEAD, KV)
  row("attention, a draft pass's 6 rows over 8192 bf16 positions", seconds(lambda: q.attend(k, v, places, fidelity=4).realize()),
      2 * KV * keys * HEAD * 2, "GB/s of the cache", dram)
  width = 2 * KH * HD + VH * HD  # Gated DeltaNet's decode step for one token: the states are read and written once
  S, x = zeros(HD, HD, VH, "fp32"), zeros(1, width + VH * HD + 2 * VH, 1, "fp32")
  conv, taps, dt, decay = zeros(1, width, 3, "fp32"), zeros(1, width, 4, "fp32"), zeros(1, VH, 1, "fp32"), zeros(1, VH, 1, "fp32")
  live = Tensor.of(rt, [[1.0]])
  row("Gated DeltaNet, a decode step", seconds(lambda: qwen.delta(S, x, conv, taps, dt, decay, live).realize()),
      2 * VH * HD * HD * 4, "GB/s of the states")
  for m in (32, 64):  # and by the chunked rule, from tensor operations, as a prompt's piece or a tree of drafts runs it
    (q, k), (v, beta, g) = (zeros(m, HD, KH, "fp32") for _ in range(2)), (zeros(m, c, VH, "fp32") for c in (HD, 1, 1))
    row(f"Gated DeltaNet, {m} tokens by the chunked rule", seconds(lambda: [t.realize() for t in qwen.chunk(q, k, v, beta, g, S)]))
  (x, y), w = (zeros(32, DIM, 1, "fp32") for _ in range(2)), zeros(1, DIM)  # the fp32 residual stream and a block's output
  row(f"RMSNorm of the stream and a block's output, 32 rows of {DIM}",
      seconds(lambda: [t.realize() for t in llama.rms(x, w, SimpleNamespace(eps=1e-6, centered=True), y)]))
  # sampling's top-k at the served 20, and a draft pass's best 4 (the tree's branch) of its 96k-token head
  for m, cols, k, what in ((32, VOCAB, 20, "sampling's top-k"), (6, 96 * 1024, 4, "a draft pass's top-k")):
    logits = np.random.default_rng(0).normal(0, 4, (m, cols)).astype(np.float32)
    z = Tensor.frombytes(rt, m, cols, logits.tobytes(), 1, "fp32")
    row(f"{what}, the {k} largest of {m} rows of {cols} logits", seconds(lambda: [t.realize() for t in serve.grouped_topk(z, k)]))
  return rows

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  if isinstance(chip.pci, pci.Sim): sys.exit("baretile.models.bench times a card; ttsim's instructions take no time")
  probe = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
  rows = table(chip, probe)
  for what, us, rate, unit, pct in rows:
    print(f"  {what:66s} {us:9.1f} us" + (f" {rate:8.1f} {unit}" if rate is not None else "") + (f" ({pct:.0f}%)" if pct is not None else ""))
  if len(sys.argv) > 2:
    with open(sys.argv[2], "w") as f: json.dump([dict(zip(("what", "us", "rate", "unit", "percent"), r)) for r in rows], f, indent=1)
