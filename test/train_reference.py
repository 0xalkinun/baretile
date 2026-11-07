# PyTorch's run of baretile.models.train's recipe, from the same initial weights and the same sequences, for #9's check
# that the card's first steps follow PyTorch's loss curve, and the comparison of two runs' logs.
#   python3 test/train_reference.py train.bin [train.bin ...] val.bin log.jsonl [key=value ...]
#   python3 test/train_reference.py compare baretile.jsonl pytorch.jsonl
# The shards, the settings and the log's lines are baretile.models.train's. stop=n ends the run after n steps of a
# schedule still computed over all of steps, so the first 100 steps of the first launch's run are precision=bf16
# steps=1907 stop=100. bf16 is PyTorch's autocast, which test_train_gpt2 checks baretile's bf16 against. Each
# sequence's share of the loss goes backward on its own, as in baretile, and AdamW is torch.optim.AdamW, decaying
# what baretile's decays, after torch.nn.utils.clip_grad_norm_. It runs on CUDA if PyTorch has it, else on the CPU.
# compare prints each step's loss in both logs and the difference, then the largest difference. A resumed run can log a
# step twice, and the last line counts.
import json, math, os, sys, time
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile.models import gpt2, train

def run(paths, c, log):
  """Train from train.init(c, c.seed) on the shards in paths, the last of which is the validation shard, for c.stop
  steps, writing a line to log for each, as baretile.models.train does."""
  import torch
  F, device = torch.nn.functional, "cuda" if torch.cuda.is_available() else "cpu"
  w = {k: torch.tensor(a, device=device, requires_grad=True) for k, a in train.init(c, c.seed).items()}
  mask = torch.tensor(gpt2.mask(range(c.ctx), c.ctx), device=device)
  groups = [{"params": [p for k, p in w.items() if train.decays(k) == d], "weight_decay": c.decay if d else 0} for d in (True, False)]
  adamw = torch.optim.AdamW(groups, lr=c.lr, betas=(c.b1, c.b2), eps=c.eps, foreach=False)
  runs, val = train.sequences([train.shard(p) for p in paths[:-1]], c.ctx), train.shard(paths[-1])
  def loss(seq):  # the mean loss over the sequence's positions
    with torch.autocast(device, torch.bfloat16, enabled=c.precision == "bf16"):
      return F.cross_entropy(train.forward(seq[:-1].tolist(), w, mask, c), torch.tensor(seq[1:], dtype=torch.int64, device=device))
  with open(log, "w") as out:
    for step in range(1, c.stop + 1):
      start, lr, total = time.time(), train.rate(step, c), 0
      for group in adamw.param_groups: group["lr"] = lr
      for _ in range(c.accum):
        share = loss(next(runs)) / c.accum
        share.backward()
        total += share.detach()
      norm = torch.nn.utils.clip_grad_norm_(list(w.values()), c.clip or math.inf).item()  # the norm before clipping
      adamw.step()
      adamw.zero_grad()
      record = dict(step=step, loss=total.item(), norm=norm, lr=lr)
      if step % c.val_every == 0 or step == c.steps:
        with torch.no_grad():
          runs_val = train.sequences([val], c.ctx)
          record["val"] = sum(loss(next(runs_val)).item() for _ in range(c.val_batches)) / c.val_batches
      took = time.time() - start
      out.write(json.dumps(record | dict(seconds=took)) + "\n")
      out.flush()
      print(f"step {step}: loss {record['loss']:.4f}, lr {lr:.2e}, gradient norm {norm:.3f}"
            + (f", validation loss {record['val']:.4f}" if "val" in record else "") + f" ({took:.1f}s on {device})", flush=True)

def records(path): return {r["step"]: r for r in map(json.loads, open(path))}  # each step's last line

def compare(ours, theirs):  # each step's loss in both logs and the difference, then the largest. Returns that difference
  a, b = records(ours), records(theirs)
  steps = sorted(a.keys() & b.keys())
  if not steps: sys.exit(f"{ours} and {theirs} have no step in common")
  print("step     ours   theirs   difference")
  for s in steps: print(f"{s:4d}   {a[s]['loss']:.4f}   {b[s]['loss']:.4f}   {a[s]['loss'] - b[s]['loss']:+.4f}")
  worst = max(steps, key=lambda s: abs(a[s]["loss"] - b[s]["loss"]))
  print(f"{len(steps)} steps in both. The largest difference is {a[worst]['loss'] - b[worst]['loss']:+.4f}, at step {worst}")
  return a[worst]["loss"] - b[worst]["loss"]

def main():
  args = sys.argv[1:]
  if args[:1] == ["compare"] and len(args) == 3: return compare(*args[1:])
  paths = [a for a in args if "=" not in a]
  if len(paths) < 3: sys.exit("usage: python3 test/train_reference.py train.bin [train.bin ...] val.bin log.jsonl [key=value ...]\n"
                             "       python3 test/train_reference.py compare baretile.jsonl pytorch.jsonl")
  c = SimpleNamespace(**train.GPT2 | dict(stop=0))
  for a in args:
    if "=" in a:
      k, v = a.split("=", 1)
      if k not in vars(c): sys.exit(f"unknown setting {k}. Settings: {', '.join(vars(c))}")
      setattr(c, k, type(getattr(c, k))(v))
  c.stop = c.stop or c.steps
  run(paths[:-1], c, paths[-1])

if __name__ == "__main__": main()
