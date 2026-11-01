# Train a small character-level GPT-2 on README and the docs, written in tinygrad: llm.c's GPT (tinygrad's
# examples/llm.c) and tinygrad's Adam, on baretile through the adapter, under TinyJit. PyTorch checks the loss and
# gradients at each step on the same weights, with baretile.models.train's model. A separate PyTorch run starts from the
# same weights to compare loss curves; floating-point rounding lets them diverge. It needs a tinygrad checkout, for
# examples/llm.c, at the commit CI pins, on PYTHONPATH or in TINYGRAD.
#   python3 test/train.py [steps]
import os, pathlib, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if os.environ.get("TINYGRAD"): sys.path.insert(0, os.environ["TINYGRAD"])
from baretile.models import gpt2
from baretile.models.train import forward, init

def text():  # README.md, then the pages in docs/ by name
  root = pathlib.Path(__file__).parent.parent
  return "".join(p.read_text() for p in [root / "README.md", *sorted((root / "docs").glob("*.md"))])

B1, B2, EPS = 0.9, 0.999, 1e-8  # Adam's defaults in PyTorch
LR = 3e-3
LINEAR = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")  # (out, in) in llm.c's GPT

def train(steps, log=None, data=None):
  """Train for steps steps on baretile and compare each step with PyTorch on the same weights.
  Also train an independent PyTorch copy from the same initial weights. Small rounding differences
  can make the two loss curves diverge: Adam's first update depends on the gradient's sign, which
  matmul error can change for gradients near zero.

  Return a tuple per step: baretile loss, PyTorch loss on the same weights, maximum relative gradient
  error, and the independent PyTorch loss. Normalize each weight's gradient error by its largest
  reference gradient. data defaults to the text of README.md and docs/."""
  import torch
  import tinygrad, baretile.adapters.tinygrad_device  # noqa: F401, the BARETILE device
  from tinygrad import Tensor, TinyJit, nn
  from tinygrad.helpers import Context
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(tinygrad.__file__)), "examples", "llm.c"))
  from train_gpt2 import GPT, GPTConfig
  data = data or text()
  chars = sorted(set(data))
  c = SimpleNamespace(ctx=64, dim=64, heads=2, layers=2)
  c.vocab, ids, rng = len(chars), np.array([chars.index(ch) for ch in data]), np.random.default_rng(1)
  w = init(c)
  # llm.c's GPT holds w with its Linear weights as (out, in) where w's are (in, out), and its biases and norms as vectors
  ours = lambda k, a: a.T if k.endswith(LINEAR) else a.reshape(-1) if a.shape[0] == 1 else a
  with Context(DEV="BARETILE"):  # the model, its optimizer and the positions it makes, on baretile
    model = GPT(GPTConfig(block_size=c.ctx, vocab_size=c.vocab, padded_vocab_size=c.vocab, n_layer=c.layers, n_head=c.heads,
                          n_embd=c.dim))
    nn.state.load_state_dict(model, {k: Tensor(np.ascontiguousarray(ours(k, a))) for k, a in w.items()}, strict=False, verbose=False)
    tw = {k: t for k, t in nn.state.get_state_dict(model).items() if k in w}  # tied, lm_head.weight is wte.weight
    opt = nn.optim.Adam(list(tw.values()), lr=LR, b1=B1, b2=B2, eps=EPS)
    @TinyJit  # records on the second call and replays after that, so the host only copies the inputs in
    def train_step(x, y):  # the loss and the gradients, after Adam updates the weights
      with Context(TRAINING=1):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        return loss.realize(*[t.grad for t in tw.values()], *opt.schedule_step()), *[t.grad for t in tw.values()]
    values = lambda k, t: ours(k, t.numpy().reshape(w[k].shape) if t.ndim == 1 else t.numpy())  # as w holds it
    mask = gpt2.mask(range(c.ctx), c.ctx)
    pw = {k: torch.tensor(a, requires_grad=True) for k, a in w.items()}
    popt = torch.optim.Adam(pw.values(), lr=LR, betas=(B1, B2), eps=EPS)
    pmask, losses = torch.tensor(mask), []
    for step in range(1, steps + 1):
      start = time.time()
      at = int(rng.integers(0, len(ids) - c.ctx - 1))
      x, y = ids[at:at + c.ctx], ids[at + 1:at + c.ctx + 1]
      same = {k: torch.tensor(values(k, t), requires_grad=True) for k, t in tw.items()}  # PyTorch on baretile's current weights
      loss, *grads = train_step(*(Tensor(a.reshape(1, -1).astype(np.int32), device="BARETILE") for a in (x, y)))
      want = torch.nn.functional.cross_entropy(forward(x.tolist(), same, pmask, c), torch.tensor(y))
      want.backward()
      error = max(float(np.abs(values(k, g) - same[k].grad.numpy()).max() / np.abs(same[k].grad.numpy()).max())
                  for k, g in zip(tw, grads))
      popt.zero_grad()
      own = torch.nn.functional.cross_entropy(forward(x.tolist(), pw, pmask, c), torch.tensor(y))
      own.backward()
      popt.step()
      losses.append((loss.item(), want.item(), error, own.item()))
      if log: log(f"step {step}: loss {losses[-1][0]:.4f}, PyTorch's {want.item():.4f} on the same weights (gradients within {error:.1e}), "
                  f"{own.item():.4f} in its own run ({time.time() - start:.1f}s)")
  return losses


def main():
  losses = train(int(sys.argv[1]) if len(sys.argv) > 1 else 300, log=print)
  mean = lambda i, part: sum(step[i] for step in part) / len(part)
  print(f"at worst, the loss was within {max(abs(f - p) for f, p, _, _ in losses):.0e} of PyTorch's on the same weights and the gradients within "
        f"{max(g for _, _, g, _ in losses):.0e}")
  n = min(50, len(losses) // 2) or 1  # compare the mean loss of the first n steps with the last n
  print(f"mean loss of the first and last {n} steps: {mean(0, losses[:n]):.2f} -> {mean(0, losses[-n:]):.2f}, PyTorch's own run "
        f"{mean(3, losses[:n]):.2f} -> {mean(3, losses[-n:]):.2f}")

if __name__ == "__main__": main()
