# Extensions to the Llama model in llama.py: mixtures of experts, Gated DeltaNet and multi-token
# prediction. The model operations run with baretile tensors or NumPy arrays. Kernels used only by
# these models live here: delta updates recurrent state, group sorts a long prompt's tokens by expert,
# and touched lists the experts that a decode step's or a short prompt's tokens selected. gpt-oss
# weights are converted from MXFP4.
import functools

import numpy as np
from baretile import bh
from baretile.runtime import DATA, ZEROS, compile_kernel
from baretile.tensix import (ACROSS, BANKS, COPIES, DST_FREE, DST_HALF, FILLED, FOLDS, MATH_END, MINUS_ONE, ONE, OUTS, RAW, REPORT, ROWS, SEM_L1,
                            SEM_PACKED, SFPU, SFPU_START, SYNC_WORDS, UNPACKABLE, ZERO, Loop, aim, allocate, brisc_program, fill, haloize, ins,
                            kernel_config, mad, mask_columns, mask_rows, math_begin, math_program, matmul, mul, pack, pack_program, reconfigure,
                            sfp_const, sfp_load, sfp_store, sfpu, slot_base, tile_of, tt, unpack, unpack_program, walk)
from baretile.tensor import Tensor
from baretile.models import gpt2, llama

def deltanet(h, p, c, state, phases, live=None, accept=None, rule=None):
  """Apply Qwen3.5/3.6's Gated DeltaNet to this call's tokens and update recurrent state in place.
  Tokens belong to one sequence, or form equal-length groups for several sequences. live marks active
  tokens; inactive tokens leave state unchanged. After the projections, computation uses fp32.
  Decode and verification run the convolution, gates and recurrence in one kernel. Prompts use
  64-token chunks.

  state contains [conv, S]:
  - conv holds the previous c.conv - 1 convolution inputs, with one matrix per history position and
    one row per sequence.
  - S holds a c.kd x c.vd matrix per value head, with c.vh batch entries per sequence. Each token
    updates it as S = e^g S + beta k (v - e^g S^T k)^T, then reads the result with its query.

  Speculative verification processes a token and its draft. state[2:] keeps copies after the first
  token. On the next call, accept=1 resumes from the state after both tokens; accept=0 resumes from
  the copies before the rejected draft.

  rule, a recurrence of the caller's, takes the place of the conv and the gated delta rule, for tokens in
  an order of its own, and the states stay as they were. rule(rows, last, taps, gates, S) gets this call's conv
  inputs (rows of x, each sequence's in turn), the last c.conv - 1 of each sequence as the conv state holds them (as
  conv takes them), the conv taps, gates, which takes the conv's output (with its SiLU, as conv gives it) to q, k, v,
  b (before its sigmoid) and g with a batch entry for each head of each sequence, and qwen.rule's raw, and S. It
  returns the outputs as chunk does, a batch entry for each value head of each sequence, and whether they come
  RMS-normed and gated already, as qwen.rule gives them with raw.

  p contains qkv_proj (q, k, v, z, b and a concatenated), conv taps, decay (-exp(A_log)), dt_bias,
  norm and o_proj."""
  f, (past, S) = llama.fns(h), state[:2]
  past = past.get() if isinstance(past, Row) else past  # the conv state of one sequence out of several
  n, seqs, qk, vw = h.shape[-2], past.shape[1], c.kh * c.kd, c.vh * c.vd
  m, width, taps = n // seqs, 2 * qk + vw, p["conv"]  # tokens per sequence, channels of the conv, and its taps (a batch entry each)
  x = f.mm(h, p["qkv_proj"], phases("qkv"), "fp32")
  alive = lambda: live.reshape(1, seqs, m)[:, :, :1]  # the first token of each sequence, with sequences as rows
  keep = lambda old, new: new if live is None else old + alive() * (new - old)
  big = lambda old, new: new if live is None else old + f.repeat(alive().reshape(seqs, 1, 1), c.vh) * (new - old)
  def each(t, d):  # a batch entry for each head of each sequence
    heads = f.split(t, t.shape[-1] // d) if d > 1 else f.heads(t)
    return heads if seqs == 1 else f.swap(heads.reshape(heads.shape[0], seqs, m * d)).reshape(seqs * heads.shape[0], m, d)
  def gates(y):  # from the conv's output: q and k of each key head (rule takes them to length 1), v of each value head,
    # b and g, and qwen.rule's raw
    y = y.reshape(n, width)
    q, k = each(y[:, :qk], c.kd), each(y[:, qk:2 * qk], c.kd)
    b, a = x[:, width + vw:width + vw + c.vh], x[:, width + vw + c.vh:] + p["dt_bias"]
    g = each(p["decay"] * (f.maximum(a, 0) + f.log(1 + f.exp(-f.maximum(a, -a)))), 1)  # softplus: the log decay of each head
    raw = b, x[:, width + vw + c.vh:], p["dt_bias"], p["decay"], x[:, width:width + vw], p["norm"], c.eps
    return q, k, each(y[:, 2 * qk:], c.vd), each(b, 1), g, raw
  done = False  # the outputs normed and gated already
  if rule is not None:  # the caller's recurrence, from the conv inputs: this call's, and the last ones where they lie
    o, done = rule(x[:, :width], past, taps, gates, S)
    o = o if seqs == 1 else f.swap(o.reshape(seqs, c.vh, m * c.vd)).reshape(c.vh, n, c.vd)
  elif (m == 1 and accept is None or m == 2 and accept is not None) and not isinstance(state[0], Row):
    # A decode step's token, or a verify step's token and then its draft, in one kernel with a unit for each key head
    # of each sequence (f.delta).
    # The kernel reads the rows of x as they are and runs the conv, the gates and the recurrence. It writes the conv
    # inputs and the states in place. In a verify step, accept says whether to go on from the state after the last
    # draft or after the token before it. The state after the first token goes into the copies in state[2:].
    o = f.delta(S, x, past, taps, p["dt_bias"], p["decay"], live, *((state[3], accept, state[2]) if m == 2 else ()))
  else:  # a prompt, maybe in one slot of several. chunk batches its pieces of 64, then runs the state through them in turn
    # The conv inputs of each sequence are its last ones, then this call's. Realize them before the last ones are
    # written over.
    rows = f.now(f.cat(f.swap(past), x[:, :width].reshape(seqs, m, width)))
    y = conv(rows.reshape(seqs * (m + c.conv - 1), width), shifted(h, seqs, m, c.conv), taps)  # each token after its last inputs
    f.assign(state[0], keep(state[0], f.swap(rows[:, m:, :])))
    q, k, v, b, g, _ = gates(y)
    o, S2 = chunk(q, k, v, f.sigmoid(b), g, S)
    o = f.now(o)  # o reads the state, so realize it before the state is written over
    f.assign(state[1], big(state[1], S2))
    o = o if seqs == 1 else f.swap(o.reshape(seqs, c.vh, m * c.vd)).reshape(c.vh, n, c.vd)
  if not done:  # each head RMS-normed and gated by z
    o = o * f.rsqrt(f.mean(o * o) + c.eps) * p["norm"] * f.silu(f.split(x[:, width:width + vw], c.vh))
  return f.mm(f.merge(f.bf16(o)), p["o_proj"], phases("o"), "fp32")

class Row:
  """View row u across a batch of matrices, one row per sequence. llama.slot uses this for DeltaNet convolution
  history. Tensor views can't select this row directly, so reads use a move and writes use put."""
  def __init__(self, t, u): self.t, self.u, self.at, self.batch, self.rows, self.cols = t, u, Tensor.of(t.rt, [[u]], "int32"), t.batch, 1, t.cols
  def get(self): return self.t[:, self.u:self.u + 1, :]
  def assign(self, x): self.t.put(self.at, x.get() if isinstance(x, Row) else x); return self
  def tobytes(self): return self.get().tobytes()

def numpy_delta(S, x, conv, taps, dt, decay, live=None, first=None, accept=None, conv2=None):
  """Run delta with NumPy for llama.fns, using the same arguments, outputs and in-place state updates."""
  n, width = conv.shape[1], conv.shape[-1]
  (vh, kd, vd), m = (S.shape[0] // n, *S.shape[1:]), x.shape[-2] // n
  qk, vw, x = (width - vh * vd) // 2, vh * vd, np.asarray(x).reshape(n, m, -1)
  each = lambda a: np.ones(n, bool) if a is None else np.asarray(a).reshape(n, -1)[:, 0] != 0  # by sequence, from the first token
  alive, taken = each(live), each(accept) if m > 1 else np.ones(n, bool)
  chain = np.where(taken[None, :, None], conv, conv2) if m > 1 else conv  # the last inputs, oldest first
  state = np.where(np.repeat(taken, vh)[:, None, None], S, first) if m > 1 else S
  unit, out = lambda a: a / np.sqrt((a * a).sum(-1, keepdims=True) + 1e-6), []
  for t in range(m):
    xt = x[:, t]
    inputs = np.concatenate([chain, xt[None, :, :width]])
    y, chain = (inputs * taps).sum(0), inputs[1:]
    y = y / (1 + np.exp(-y))
    q, k = unit(y[:, :qk].reshape(n, -1, kd)) * kd ** -0.5, unit(y[:, qk:2 * qk].reshape(n, -1, kd))
    b, a = xt[:, 2 * qk + 2 * vw:2 * qk + 2 * vw + vh], xt[:, 2 * qk + 2 * vw + vh:] + np.reshape(dt, vh)
    beta, g = 1 / (1 + np.exp(-b)), np.reshape(decay, vh) * (np.maximum(a, 0) + np.log1p(np.exp(-np.abs(a))))
    column = lambda a: np.repeat(a, vh // a.shape[1], 1).reshape(n * vh, kd, 1)  # the key head of each value head, as a column
    row = lambda a: a.reshape(n * vh, 1, -1)  # each value head as a row
    o, state = step(column(q), column(k), row(y[:, 2 * qk:]), row(beta), row(g), state)
    (into, states), heads = (conv2, first) if t < m - 1 else (conv, S), np.repeat(alive, vh)
    into[:, alive], states[heads] = chain[:, alive], state[heads]
    out.append(o.reshape(n, vh, vd) * alive[:, None, None])
  return np.stack(out, 2).transpose(1, 0, 2, 3).reshape(vh, n * m, vd)

def step(q, k, v, beta, g, S):
  """Apply the gated delta rule to one token in NumPy, matching transformers' torch_recurrent_gated_delta_rule.
  q and k are columns. Return (output, updated state)."""
  e = np.exp(g)
  v = beta * (v - e * (k * S).sum(-2, keepdims=True))  # correct the decayed state's prediction toward v
  S = e * S + k * v
  return (q * S).sum(-2, keepdims=True), S

def corrected(k, v, beta, g, seen=None):
  """The part of a piece's gated delta rule that doesn't read the state (chunk, or a deltanet rule). seen[i, j] is 1
  where token i follows token j or is it: the causal triangle when None, or another order, such as a tree's ancestors,
  with each token after the ones it follows. Return cum, the log decay through each token, the decay from token j to
  token i where i follows j (else 0), and u and w, which solve (I + L) u = beta v and (I + L) w = beta k e^cum. L is
  strictly lower triangular, since a token comes after the ones it follows, and describes how each token's correction
  affects the ones after it. Its inverse is built by joining blocks, from one token up: two inverted blocks A^-1 and
  B^-1, with C below the diagonal between them, join as [A^-1 0; -B^-1 C A^-1, B^-1], two matmuls for every pair of
  blocks at once. Each step reads only inverses, which stay near 1. A product (I - L)(I + L^2)(I + L^4)... over the
  whole piece is the same inverse, but its powers of L reach 1e10 where a piece's keys point nearly the same way, as
  in Qwen3.5-9B's later blocks, and float32 loses the result in their cancellation."""
  f, s = llama.fns(k), k.shape[-2]
  upper = f.tri(s)
  cum = f.t(f.t(g) @ (upper if seen is None else f.t(seen)))  # the log decay from the start of the piece, along its path
  decay = f.exp(f.minimum(cum - f.t(cum), 0)) * (f.tri(s, True) if seen is None else seen)
  kbeta = k * beta
  L = (kbeta @ f.t(k)) * decay * (1 - upper)
  inv, span = upper * f.tri(s, True) - L * f.pairs(s, 1), 2  # pairs of single tokens: [1 0; l 1]^-1 is [1 0; -l 1]
  while span < s:
    inv, span = inv - (inv @ (L * f.pairs(s, span))) @ inv, 2 * span
  return cum, decay, inv @ (v * beta), inv @ (kbeta * f.exp(cum))

def chunk(q, k, v, beta, g, S, size=64):
  """Apply the gated delta rule to a prompt in 64-token pieces, following transformers'
  torch_chunk_gated_delta_rule. Take initial state S, queries q and keys k for each key head (as rule), values v,
  and columns beta and g. Return each token's output and the final state. Pad the last piece with tokens whose key, value
  and decay are zero, so they don't change state. Each piece runs by rule, the state through them in turn, with q and k
  taken to length 1 once."""
  f, (h, m, dv) = llama.fns(q), v.shape
  n, s = -(-m // size), min(m, size)  # the number of pieces, and the tokens in each
  q, k = unit(q, q.shape[-1] ** -0.5), unit(k)
  if n == 1: return (lambda o, vnew, cum, S: (o, S))(*rule(q, k, v, beta, g, S, state=True, normed=True))
  # copy each, since it may be a repeat or a view, and pad to a multiple of 64. piece i of head j is batch entry j n + i
  q, k, v, beta, g = (f.parts(f.pad(t, n * s - m) if n * s > m else t + 0, n) for t in (q, k, v, beta, g))
  out = f.empty((h, n * s, dv))
  for i in range(n):
    o, _, _, S = rule(*(t[i::n, :, :] for t in (q, k, v, beta, g)), S, state=True, normed=True)
    f.assign(f.parts(out, n)[i::n, :, :], o)
  return out if n * s == m else out[:, :m, :], S

def delta_kernel(kt, vt, vh, rep, m=1):
  """The kernel for delta: the Gated DeltaNet decode step of each sequence, or a verify step's token and its draft,
  from the projections of the tokens to their outputs. It's fp32 throughout and reads each state once.

  Arguments: the first unit and the number of units, then four words for each tensor (DRAM address, batch tile
  stride, row tile stride, tile offset), in this order:
  - S, the states
  - the output: a matrix for each head, with the m tokens of each sequence as rows in turn
  - live: bf16, a row for each token
  - with m = 2: first (the states after the first token), and accept (a column for each sequence)
  - x: a row for each token with q, k, v, z, b and a side by side, the m tokens of a sequence in turn
  - conv: the last 3 inputs of the conv, oldest first, with a row for each sequence. With m = 2, conv2 follows: the
    same after the first token.
  - taps: 4 matrices of one row
  - dt and decay: a column for each value head

  A unit is a key head of a sequence and its rep value heads, so no two units read or write the same channels of the
  conv inputs. BRISC reads the states of the unit, and the runs of its channels from the conv inputs, the rows of x
  and the taps. q and k go into columns 0-3 of a tile with a row for each channel. v goes into rows 0-3 with a column
  for each channel. b, a, dt and decay go at position 0. Once BRISC has read the inputs, it writes back the ones that
  each token leaves: they move up a position, and the token's input goes last. With m = 2 those after the first token
  go into conv2. BRISC skips this for a sequence that isn't live.

  The math thread masks everything outside those positions to 0 in Dst, so no tile in L1 needs zeroing. Then it folds.
  The 4 inputs of a channel times its 4 taps are summed across the row (or down the column for v), then go through
  SiLU into a broadcast tile. q and k go to length 1 by the sum of their squares, folded down, and q is scaled by
  kd^-1/2. beta = sigmoid(b) and e^g = exp(decay softplus(a + dt)).
  The rule then splits by column. For each value head, each column tile j of its S, and each token in turn, one SFPU
  pass sums u = S^T k and w = S^T q down the tiles of the column. Then d = beta (v - e^g u), the output of the token
  is e^g w + d (k . q), and the new tiles of the column are e^g S + k d, which is the S of the next token.
  Write the final states to S. With two tokens, save the states after the first token in first. For an inactive
  sequence, BRISC reads only live and accept, leaves the states unchanged, and writes zero outputs."""
  kh, kd, vd = vh // rep, 32 * kt, 32 * vt
  qk, vw = kh * kd, vh * vd
  S_ = lambda e, i, j: f"(F0 + ({e}) * {kt * vt} + ({i}) * {vt} + ({j}))"  # tile (i, j) of the S of value head e
  T0 = rep * kt * vt  # then the taps for q and k (in columns) and for v (in rows), then dt and decay
  TQ, TK = (lambda i, o=o: f"(F0 + {T0 + o} + ({i}))" for o in (0, kt))
  TV = lambda e, j: f"(F0 + {T0 + 2 * kt} + ({e}) * {vt} + ({j}))"
  DT, DC = (lambda e, o=o: f"(F0 + {T0 + 2 * kt + rep * vt + o} + ({e}))" for o in (0, rep))
  per = 2 * kt + rep * vt + 2 * rep  # tiles of the taps, dt and decay. Each token has as many: its q, k and v inputs, b and a
  I0 = T0 + per  # then the tiles of each token
  IQ, IK = (lambda t, i, o=o: f"(F0 + {I0 + o} + ({t}) * {per} + ({i}))" for o in (0, kt))
  IV = lambda t, e, j: f"(F0 + {I0 + 2 * kt} + ({t}) * {per} + ({e}) * {vt} + ({j}))"
  BB, AA = (lambda t, e, o=o: f"(F0 + {I0 + 2 * kt + rep * vt + o} + ({t}) * {per} + ({e}))" for o in (0, rep))
  F = I0 + m * per  # slots in a frame
  mine = per + 1  # the broadcast tiles of a token, packed: k, q and v, then e^g, beta and k . q
  # What the packer writes comes first, after slot 0, since it reaches only bh.PACK_REACH past slot 0. The frames follow.
  KC, QC, VV, E, BT, KQ = (lambda t, o=o: 1 + t * mine + o
                           for o in (0, kt, 2 * kt, 2 * kt + rep * vt, 2 * kt + rep * vt + rep, mine - 1))
  COL = 1 + m * mine  # the new tiles of the column after the first of two tokens: the S of the second
  OUT = COL + kt * (m - 1)  # an output slot for each Dst half
  FR = OUT + 2  # the first frame
  sources = 7 + m  # the rows of the conv that a unit reads: the 3 last inputs, one for each token, and the 4 taps
  STAGE = slot_base("fp32") + (FR + 2 * F) * 4096  # BRISC stages the q and k runs of each source here, then a 64-byte block per scalar
  scalars = 2 + 2 * rep + 2 * m * rep  # live and accept, dt and decay, and b and a for each token
  if STAGE + sources * 256 * kt + 64 * scalars > OUTS or FR + 2 * F > UNPACKABLE or FR * 4096 > bh.PACK_REACH:
    raise ValueError("delta_kernel: a unit's tiles don't fit in L1")
  # Dst rows, 64 to a tile in fp32. R2 and R3 take fills, R4 and R5 the two sums (or a norm, or decay), R6 takes e^g,
  # and R7 takes d or x set aside.
  R2, R3, R4, R5, R6, R7 = 128, 192, 256, 320, 384, 448
  programs = {  # each runs at Dst row 0 (aim(0)) and stores its result to row r
    "kq first": lambda r: [sfp_load(0, R2), sfp_load(1, R3), mul(2, 0, 1), sfp_store(2, R4)],
    "kq": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R4), mad(2, 0, 1, 2), sfp_store(2, R4)],
    "sums first": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), mul(3, 1, 0), mul(4, 2, 0), sfp_store(3, R4, False),
                             sfp_store(4, R5)],
    "sums": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), sfp_load(3, R4), sfp_load(4, R5), mad(3, 1, 0, 3),
                       mad(4, 2, 0, 4), sfp_store(3, R4, False), sfp_store(4, R5)],
    "d": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), sfp_load(3, R4), mad(4, 2, 3, 1, neg=1), mul(5, 0, 4),
                    sfp_store(5, R7)],  # beta (v - e u)
    "out": lambda r: [sfp_load(0, R6), sfp_load(1, R5), sfp_load(2, R7), sfp_load(3, R2), mul(4, 0, 1), mad(5, 2, 3, 4),
                      sfp_store(5, r)],  # e w + d (k . q)
    "state": lambda r: [sfp_load(0, R2), sfp_load(1, R3), sfp_load(2, R6), sfp_load(3, R7), mul(4, 2, 0), mad(5, 1, 3, 4),
                        sfp_store(5, r)],  # e S + k d
    "copy": lambda r: [sfp_load(0, R4), sfp_store(0, r)],
    "mul": lambda r: [sfp_load(0, R2), sfp_load(1, R3), mul(2, 0, 1), sfp_store(2, r)],  # inputs times their taps
    "add": lambda r: [sfp_load(0, R2), sfp_load(1, R3), mad(2, 0, ONE, 1), sfp_store(2, r)],  # a + dt
    "aside": lambda r: [sfp_load(0, r), sfp_store(0, R7)],  # keeps x for the x sigmoid(x) of SiLU and the max(x, 0) of softplus
    "neg": lambda r: [sfp_load(0, r), mul(1, 0, MINUS_ONE), sfp_store(1, r)],
    "-abs": lambda r: [sfp_load(0, r), ins("SFPABS", vd=1, vc=0, mod1=1), mul(2, 1, MINUS_ONE), sfp_store(2, r)],
    "1+": lambda r: [sfp_load(0, r), mad(1, 0, ONE, ONE), sfp_store(1, r)],
    "times x": lambda r: [sfp_load(0, r), sfp_load(1, R7), mul(2, 0, 1), sfp_store(2, r)],
    "+ max(x, 0)": lambda r: [sfp_load(0, R7), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_load(2, r),
                              mad(3, 2, ONE, 0), sfp_store(3, r)],
    "times R4": lambda r: [sfp_load(0, r), sfp_load(1, R4), mul(2, 0, 1), sfp_store(2, r)],  # times decay, or q or k times its scale
    "squares first": lambda r: [sfp_load(0, R2), mul(1, 0, 0), sfp_store(1, R4)],
    "squares": lambda r: [sfp_load(0, R2), sfp_load(1, R4), mad(1, 0, 0, 1), sfp_store(1, R4)],
    "+ eps": lambda r: [sfp_load(0, R4), *sfp_const(5, 1e-6), mad(1, 0, ONE, 5), sfp_store(1, R4)],
    "q's scale": lambda r: [sfp_load(0, R4), *sfp_const(5, kd ** -0.5), mul(1, 0, 5), sfp_store(1, R4)]}
  # The programs, folds, masks and SFPU programs that the math thread runs are functions (helpers, below), so their
  # words are in the program once and the loops that call them stay short. A program takes the row for its result. The
  # rest work wherever aim_at last aimed.
  name = lambda n: "p_" + "".join(ch if ch.isalnum() else "_" for ch in n)
  call = lambda *names: " ".join(f"{name(n)}();" for n in names)
  apply = lambda n: f"{name(n)}(0);"  # for a program that stores to its own rows
  to_h = lambda n: f"{name(n)}(h);"  # for a program whose result goes to Dst half h
  aim_at = lambda row: tt(aim(row))
  free = tt([ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_SFPU)])  # fills wait until the SFPU is done with their rows
  fills = lambda *pairs: free + " " + tt([w for slot, row in pairs for w in fill(slot, row)] + FILLED)
  half = tt([DST_FREE])  # waits until a Dst half is free for a result
  done = f"{tt(MATH_END)} h ^= {DST_HALF};"
  lib = lambda *names: f"{aim_at('h')} {call(*names)} {aim_at(0)}"  # SFPU library programs, run on the tile at h
  silu = f"{to_h('aside')} {to_h('neg')} {lib('fexp')} {to_h('1+')} {lib('frecip')} {to_h('times x')}"
  masks = {"columns 4 on": mask_columns("sum", 4), "rows 4 on": mask_rows("sum", 4), "all but place 0": mask_rows("sum", 1) + mask_columns("sum", 1)}
  only = lambda mask: f"{aim_at('h')} {call(mask)}"  # zeroes the tile at h outside the positions that were filled
  conv = lambda x, taps, mask, kind: f"{aim_at(0)} {fills((x, R2), (taps, R3))} {to_h('mul')} {only(mask)} {call(kind)} {aim_at(0)} {silu}"
  # Generate (math, pack) words for Dst half h. Pack intermediates into slot and final results into the output slot
  # for this half, with `home` as the destination tile. A result used both ways stays in slot until written to DRAM;
  # keep identifies the output slot used to track its completion.
  def result(words, slot=None, home=None, keep=None):
    math = f"{half} {words} {done}" + (" posted++;" if slot is not None else "")
    if home is None: return math, f"{tt(pack('h', slot, SEM_L1))} h ^= {DST_HALF};"
    to, keep = (slot, keep) if slot is not None else (f"{OUT} + h / {DST_HALF}", f"h / {DST_HALF}")
    return math, f"output({keep}, {to}, {home}); {tt(pack('h', to, SEM_PACKED, *([SEM_L1] if slot is not None else [])))} h ^= {DST_HALF};"
  loop = lambda var, n, pair: tuple(f"for (unsigned {var} = 0; {var} < {n}; {var}++) {{ {w} }}" for w in pair)  # a (math, pack) pair, n times
  # Scales the kt tiles of q or k at slot c to length 1, and by scale if given. The sum of their squares is folded down,
  # 1 / sqrt of it goes in R4, then each tile is multiplied by it.
  def unit_length(c, scale):
    return [(f"drain(); {aim_at(0)} {fills((c, R2))} {apply('squares first')} for (unsigned i = 1; i < {kt}; i++) {{ {fills((f'{c} + i', R2))} "
             f"{apply('squares')} }} {aim_at(R4)} {call('sum1')} {aim_at(0)} {apply('+ eps')} {aim_at(R4)} {call('sqrt', 'frecip')} {aim_at(0)}"
             + (f" {apply(scale)}" if scale else ""), ""),
            loop("i", kt, result(f"{fills((f'{c} + i', 'h'))} {to_h('times R4')}", f"{c} + i"))]
  steps = []
  for t in range(m):  # the broadcast tiles of each token: the conv, the gates and k . q
    steps += [loop("i", kt, result(conv(IQ(t, "i"), TQ("i"), "columns 4 on", "sum2"), f"{QC(t)} + i")),
              loop("i", kt, result(conv(IK(t, "i"), TK("i"), "columns 4 on", "sum2"), f"{KC(t)} + i")),
              *unit_length(QC(t), "q's scale"), *unit_length(KC(t), None),
              loop("e", rep, loop("j", vt, result(conv(IV(t, "e", "j"), TV("e", "j"), "rows 4 on", "sum1"), f"{VV(t)} + e * {vt} + j"))),
              loop("e", rep, result(f"{aim_at('h')} {fills((BB(t, 'e'), 'h'))} {only('all but place 0')} {call('sum2', 'sum1')} {aim_at(0)} "
                                    f"{to_h('neg')} {lib('fexp')} {to_h('1+')} {lib('frecip')}", f"{BT(t)} + e")),  # beta = sigmoid(b)
              loop("e", rep, result(f"{aim_at(0)} {fills((DC('e'), R4))} {aim_at(R4)} {call('all but place 0', 'sum2', 'sum1')} {aim_at(0)} "
                                    f"{fills((AA(t, 'e'), R2), (DT('e'), R3))} {to_h('add')} {only('all but place 0')} {call('sum2', 'sum1')} "
                                    f"{aim_at(0)} {to_h('aside')} {to_h('-abs')} {lib('fexp')} {to_h('1+')} {lib('log')} {to_h('+ max(x, 0)')} "
                                    f"{to_h('times R4')} {lib('fexp')}", f"{E(t)} + e")),  # e^g = exp(decay softplus(a + dt))
              (f"drain(); {aim_at(0)} {fills((KC(t), R2), (QC(t), R3))} {apply('kq first')} for (unsigned i = 1; i < {kt}; i++) "
               f"{{ {fills((f'{KC(t)} + i', R2), (f'{QC(t)} + i', R3))} {apply('kq')} }} {aim_at(R4)} {call('sum1')} {aim_at(0)}", ""),
              result(f"{free} {to_h('copy')}", KQ(t)), ("drain();", "")]
  column = []  # column tile j of the S of each value head, through each token in turn
  for t in range(m):
    col = (lambda i: S_("e", i, "j")) if t == 0 else (lambda i: f"{COL} + ({i})")
    column += [("drain();" if t else "", ""),
               (f"{fills((col(0), R2), (KC(t), R3), (QC(t), R6))} {apply('sums first')} "
                f"for (unsigned i = 1; i < {kt}; i++) {{ {fills((col('i'), R2), (f'{KC(t)} + i', R3), (f'{QC(t)} + i', R6))} {apply('sums')} }} "
                f"{aim_at(R4)} {call('sum1')} {aim_at(R5)} {call('sum1')} {aim_at(0)} "
                f"{fills((f'{BT(t)} + e', R2), (f'{VV(t)} + e * {vt} + j', R3), (f'{E(t)} + e', R6))} {apply('d')}", ""),
               result(f"{fills((KQ(t), R2))} {to_h('out')}", home=f"{1 << 30:#x} | (unit * {rep} + e) << 8 | {t} << 6 | j"),
               loop("i", kt, result(f"{fills((col('i'), R2), (f'{KC(t)} + i', R3))} {to_h('state')}", f"{COL} + i" if t < m - 1 else None,
                                    f"{(m - 1 - t) << 31:#x} | (unit * {rep} + e) << 8 | (i * {vt} + j)", "2 + i"))]
  steps.append(loop("e", rep, loop("j", vt, (" ".join(c for c, _ in column), " ".join(c for _, c in column)))))
  math, packs = [w for w, _ in steps], [w for _, w in steps]
  units = lambda body: f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, F0 = {FR} + (u & 1) * {F};
    {body}
  }}"""
  # For each unit, BRISC waits for its frame to be free, then puts its states, the runs of its channels and its scalars in
  # place, then writes back the updated conv inputs. It writes results to DRAM as they're packed.
  # Below are the indices in args of first (which is S when m is 1), x, the conv inputs, the taps, dt and decay.
  first, X = (14, 22) if m > 1 else (2, 14)
  C, C2, TAPS = X + 4, X + 8, X + 4 + 4 * m
  DTA, DCA = TAPS + 4, TAPS + 8
  sc = lambda k: f"STAGE + {sources * 256 * kt} + 64 * ({k})"  # the 64-byte block of scalar k
  scalar = lambda i, row, col, k, size=4: (f"{{ unsigned t_ = {tile_of(i, 0, f'({row}) / 32', f'({col}) / 32', 'A')}, "  # size is 2 for bf16
                                           f"o_ = at(({row}) % 32, ({col}) % 32) * {size}; noc_get(BANK(t_), AT(A[{i}], t_, {1024 * size}) "
                                           f"+ o_ / 64 * 64, {sc(k)}, 64); word[{k}] = o_ % 64 / {size}; }}")
  value = lambda k, size=4: f"((volatile {'unsigned short' if size == 2 else 'unsigned'} *)({sc(k)}))[word[{k}]]"
  staged = lambda src, g: f"(STAGE + ({src}) * {256 * kt} + ({g}) * {128 * kt})"  # the q (g 0) or k (g 1) runs staged from source src
  brisc = SYNC_WORDS + BANKS + f"""
#define SLOT(i) ({slot_base("fp32"):#x} + (i) * 4096)
#define STAGE {STAGE:#x}
static const unsigned *A;
static unsigned fetched, events, frame_done[2], sent, alive[4], word[{scalars}];
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // write packed output rows (zero if inactive) and active state tiles to DRAM
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1), unit = t >> 8 & 0x3FFFFF, s = unit / {vh}, h = unit % {vh};
    unsigned live = alive[unit / {rep} % 4];
    if (t >> 30 & 1) for (unsigned f = 0; f < 2; f++) {{  // an output tile, for the token of the head: the runs of its row
      unsigned r = s * {m} + (t >> 6 & 3), o = {tile_of(6, 'h', 'r / 32', 't & 63', 'A')}, off = at(r % 32, 16 * f) * 4;
      noc_put((live ? SLOT(slot) : {ZEROS:#x}) + off, BANK(o), AT(A[6], o, 4096) + off, 64);
    }} else if (live) {{  // a tile of the states, or of those after the first token
      unsigned i = t >> 31 ? {first} : 2, o = {tile_of('i', 'unit', f'(t & 255) / {vt}', f'(t & 255) % {vt}', 'A')};
      noc_put(SLOT(slot), BANK(o), AT(A[i], o, 4096), 4096);
    }}
    noc_wait();
  }}
}}
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
// Reads the row of source src: the 3 last conv inputs (from the states at A[c]), the rows of the tokens in x, or the 4
// taps. It takes the runs of the channels in n tiles from col on, from both faces that the row crosses, into L1 at to.
// Each tile goes step bytes on from the last, and the second face goes half bytes on from the first.
static void get_runs(unsigned src, unsigned s, unsigned c, unsigned col, unsigned n, unsigned to, unsigned step, unsigned half) {{
  unsigned i = src < 3 ? c : src < {3 + m} ? {X} : {TAPS}, b = src < 3 ? src : src < {3 + m} ? 0 : src - {3 + m};
  unsigned row = src < 3 ? s : src < {3 + m} ? s * {m} + src - 3 : 0, t = {tile_of('i', 'b', 'row / 32', 'col / 32', 'A')};
  unsigned off = at(row % 32, 0) * 4;
  for (unsigned j = 0; j < n; j++, t++, to += step) {{
    noc_get(BANK(t), AT(A[i], t, 4096) + off, to, 64);
    noc_get(BANK(t), AT(A[i], t, 4096) + off + 1024, to + half, 64);
  }}
}}
// The reverse: writes the runs of row s from L1 at from, laid out the same, into conv input d at A[i]
static void put_runs(unsigned i, unsigned d, unsigned s, unsigned col, unsigned n, unsigned from, unsigned step, unsigned half) {{
  unsigned t = {tile_of('i', 'd', 's / 32', 'col / 32', 'A')}, off = at(s % 32, 0) * 4;
  for (unsigned j = 0; j < n; j++, t++, from += step) {{
    noc_put(from, BANK(t), AT(A[i], t, 4096) + off, 64);
    noc_put(from + half, BANK(t), AT(A[i], t, 4096) + off + 1024, 64);
  }}
}}
void kmain(volatile unsigned *args) {{
  A = (const unsigned *)args;
  for (unsigned u = 0; u < A[1]; u++) {{
    unsigned unit = A[0] + u, F0 = {FR} + (u & 1) * {F}, s = unit / {kh}, hk = unit % {kh};
    unsigned cq = hk * {kd}, ck = {qk} + hk * {kd}, cv = {2 * qk} + hk * {rep * vd};  // its q and k channels, and the v channels of its value heads
    reached(frame_done[u & 1]);
    {scalar(10, f's * {m}', 0, 0, 2)} {scalar(18, 0, 's', 1) if m > 1 else ""}
    noc_wait();
    alive[unit % 4] = {value(0, 2)} != 0;
    if (!alive[unit % 4]) {{ ready(); frame_done[u & 1] = ++events; continue; }}  // a sequence that isn't live reads nothing
    // accept selects the state after the previous token or its draft
    unsigned taken = {value(1) if m > 1 else 1}, from = taken ? 2 : {first}, c = taken ? {C} : {C2};
    for (unsigned e = 0; e < {rep}; e++) for (unsigned i = 0; i < {kt}; i++) for (unsigned j = 0; j < {vt}; j++) {{
      unsigned t = {tile_of('from', f's * {vh} + hk * {rep} + e', 'i', 'j', 'A')};
      noc_get(BANK(t), AT(A[from], t, 4096), SLOT({S_('e', 'i', 'j')}), 4096);
    }}
    for (unsigned src = 0; src < {sources}; src++) for (unsigned g = 0; g < 2; g++)  // stage the runs of q and k
      get_runs(src, s, c, g ? ck : cq, {kt}, {staged('src', 'g')}, 128, 64);
    for (unsigned r = 0; r < 4; r++) {{  // v into rows 0-3 of its tiles, for the inputs of each token and the taps
      for (unsigned t = 0; t < {m}; t++) get_runs(t + r, s, c, cv, {rep * vt}, SLOT({IV('t', 0, 0)}) + r * 64, 4096, 1024);
      get_runs({3 + m} + r, s, c, cv, {rep * vt}, SLOT({TV(0, 0)}) + r * 64, 4096, 1024);
    }}
    for (unsigned e = 0; e < {rep}; e++) {{
      unsigned h = hk * {rep} + e;
      {scalar(DTA, 0, 'h', '2 + e')} {scalar(DCA, 0, 'h', f'{2 + rep} + e')}
      for (unsigned t = 0; t < {m}; t++) {{
        {scalar(X, f's * {m} + t', f'{2 * qk + 2 * vw} + h', f'{2 + 2 * rep} + t * {2 * rep} + e')}
        {scalar(X, f's * {m} + t', f'{2 * qk + 2 * vw + vh} + h', f'{2 + 3 * rep} + t * {2 * rep} + e')}
      }}
    }}
    noc_wait();
    for (unsigned g = 0; g < 2; g++) for (unsigned i = 0; i < {kt}; i++) {{  // q and k, a row per channel: input or tap j in column j
      unsigned *taps = (unsigned *)SLOT(g ? {TK('i')} : {TQ('i')}), *ins[{m}];
      // tile i of source 0, with channel r at word r (each source sits {64 * kt} words past the one before)
      const unsigned *from = (const unsigned *){staged(0, 'g')} + i * 32;
      for (unsigned t = 0; t < {m}; t++) ins[t] = (unsigned *)SLOT(g ? {IK('t', 'i')} : {IQ('t', 'i')});
      for (unsigned r = 0; r < 32; r++) for (unsigned j = 0; j < 4; j++) {{
        taps[at(r, j)] = from[({3 + m} + j) * {64 * kt} + r];
        for (unsigned t = 0; t < {m}; t++) ins[t][at(r, j)] = from[(t + j) * {64 * kt} + r];
      }}
    }}
    for (unsigned e = 0; e < {rep}; e++) {{
      ((volatile unsigned *)SLOT({DT('e')}))[0] = {value('2 + e')}, ((volatile unsigned *)SLOT({DC('e')}))[0] = {value(f'{2 + rep} + e')};
      for (unsigned t = 0; t < {m}; t++) {{
        ((volatile unsigned *)SLOT({BB('t', 'e')}))[0] = {value(f'{2 + 2 * rep} + t * {2 * rep} + e')};
        ((volatile unsigned *)SLOT({AA('t', 'e')}))[0] = {value(f'{2 + 3 * rep} + t * {2 * rep} + e')};
      }}
    }}
    // after reading, update convolution history for token t with sources t + 1 to t + 3
    for (unsigned t = 0; t < {m}; t++) for (unsigned d = 0; d < 3; d++) {{
      unsigned into = t == {m - 1} ? {C} : {C2}, src = t + 1 + d, tv = src > 3 ? src - 3 : 0;
      for (unsigned g = 0; g < 2; g++) put_runs(into, d, s, g ? ck : cq, {kt}, {staged('src', 'g')}, 128, 64);
      put_runs(into, d, s, cv, {rep * vt}, SLOT({IV('tv', 0, 0)}) + (src - tv) * 64, 4096, 1024);  // from the tiles of token tv
    }}
    l1_settle(SLOT(F0));
    ready(); frame_done[u & 1] = ++events;
  }}
  while (sent < A[1] * {rep * vt * m * (kt + 1)}) send();
}}"""
  def body(p):  # a program's words, its result to row 0 or the other Dst half. One that ignores r is emitted once
    other, zero = (tt(SFPU_START + [Loop(32, p(r))]) for r in (DST_HALF, 0))
    return zero if other == zero else f"if (h) {{ {other} }} else {{ {zero} }}"
  helper = lambda n, words, arg="void": f"__attribute__((noinline)) static void {name(n)}({arg}) {{ {words} }}\n"
  helpers = "".join([helper(k, tt(ACROSS[FOLDS[k]](k))) for k in ("sum1", "sum2")] +  # sum1 sums down the columns, sum2 along the rows
                    [helper(n, tt(sfpu([n], {}))) for n in ("fexp", "frecip", "log", "sqrt")] +
                    [helper(n, tt(words)) for n, words in masks.items()] + [helper(n, body(p), "unsigned h") for n, p in programs.items()])
  cfg = kernel_config("fp32", "fp32")
  return compile_kernel(brisc, unpack_program(units("fetched();"), cfg),
                        math_program({}, units("fetched(); " + " ".join(math) + " " + tt(REPORT)), True, helpers),
                        pack_program(units(" ".join(packs)), 2 + kt))

def delta(S, x, conv, taps, dt, decay, live=None, first=None, accept=None, conv2=None):
  """The Gated DeltaNet decode step in one kernel (delta_kernel), for a token of each sequence, or a verify step's
  token and its draft. It starts from the projections of the tokens, updates the states in place, and reads each state
  once. Each token does S = e^g S + beta k (v - e^g S^T k)^T, and the output of each head is S^T q after that.

  - S: the states, kd x vd, for each sequence and value head
  - x: the projections of each token in a row (q, k, v, z, b and a side by side), the tokens of a sequence in turn
  - conv: the last 3 inputs of the conv, 3 matrices with the oldest first and a row for each sequence. Each token
    moves them up a position and adds its input last.
  - taps: 4 matrices of one row. dt and decay: a row with a column for each value head.
  - live: a bf16 column with an entry for each token, all 1 if not given. Where the first token of a sequence has 0,
    its conv and states aren't read or written, and its outputs are 0.
  - first and conv2, for a token and its draft: the states and conv inputs after the first token. accept, a row
    with 1 or 0 for each sequence, says to go on from S and conv (1) or from these (0). first and conv2 then take the
    state after this step's token, and S and conv the state after its draft.

  q, k and v are the SiLU of the conv. q and k go to length 1, and q is scaled by kd^-1/2. beta = sigmoid(b) and
  g = decay softplus(a + dt). Everything but live is fp32. Returns the outputs in fp32, as (value heads, the tokens of
  each sequence in turn, vd). Head dimensions must be multiples of 32. There's no gradient."""
  n, kd, vd, width = conv.rows, S.rows, S.cols, conv.cols
  if kd % 32 or vd % 32: raise ValueError("delta needs key and value head dimensions divisible by 32")
  if x.rows % n or S.batch % n: raise ValueError("delta needs the same number of sequences in inputs and states")
  m, vh = x.rows // n, S.batch // n
  kh, extra = divmod(width - vh * vd, 2 * kd)
  if extra or kh < 1 or vh % kh: raise ValueError("delta needs complete q/k/v projections and value heads in groups of key heads")
  aux = (first, accept, conv2)
  if m not in (1, 2) or any((t is None) == (m == 2) for t in aux):
    raise ValueError("delta needs one token, or two tokens with first, accept and conv2 together")
  ins = (S, x, conv, taps, dt, decay, *(aux if m == 2 else ()))
  if not all(t.dtype == "fp32" for t in ins): raise ValueError("delta needs fp32 states and inputs")
  if any(t.rt is not S.rt for t in (*ins, *((live,) if live is not None else ()))):
    raise ValueError("delta needs states and inputs on the same runtime")
  if not (x.shape == (1, n * m, width + vh * vd + 2 * vh) and conv.batch == 3 and taps.shape == (4, 1, width)):
    raise ValueError(f"delta needs a row of q, k, v, z, b and a for each token, 3 conv inputs and 4 taps: {x.shape} {conv.shape} {taps.shape}")
  if dt.shape != (1, 1, vh) or decay.shape != dt.shape: raise ValueError(f"delta needs dt and decay as a row of {vh}")
  if live is not None and (live.shape != (1, n * m, 1) or live.dtype != "bf16"):
    raise ValueError(f"delta needs live as a bf16 column of {n * m}")
  if m == 2 and not (first.shape == S.shape and conv2.shape == conv.shape and accept.shape == (1, 1, n)):
    raise ValueError("delta needs first like S, conv2 like conv, and accept as a row of sequences")
  live = live if live is not None else S.number(1.0, "bf16").expand(1, n * m, 1).realize()
  o = Tensor.frombytes(S.rt, n * m, vd, bytes(4 * vh * n * m * vd), vh, "fp32")  # zeroed, because no unit writes its padding rows
  key = ("delta", -(-kd // 32), -(-vd // 32), vh, vh // kh, m)
  return o.run(key, lambda: delta_kernel(*key[1:]), S, o, live, *((first, accept) if m > 1 else ()), x, conv, *((conv2,) if m > 1 else ()),
               taps, dt, decay, units=n * kh, name="delta")

def helpers(programs, setups={}):
  """SFPU programs as C functions p_name(h), each over the tile at Dst half h, its words once in the program. A program
  takes the row of its result, and its setup, if any, runs before its loop."""
  body = lambda n, p, r: tt(SFPU_START + setups.get(n, []) + [Loop(32, p(r))])
  return "".join(f"__attribute__((noinline)) static void p_{n}(unsigned h) {{ if (h) {{ {body(n, p, DST_HALF)} }} "
                 f"else {{ {body(n, p, 0)} }} }}\n" for n, p in programs.items())

def shifted(x, seqs, m, n):
  """conv's picks for a prompt of m tokens in each of seqs sequences, whose conv inputs are each sequence's last n - 1
  and then its m: token i's n inputs, oldest first. An int32 tensor on x's chip, made once for each shape."""
  picks = np.int32([[b * (m + n - 1) + i + j for b in range(seqs) for i in range(m)] for j in range(n)])
  return picks if isinstance(x, np.ndarray) else _shifted(x.rt, picks.tobytes(), n, seqs * m)

@functools.cache
def _shifted(rt, picks, n, cols): return Tensor.frombytes(rt, 1, cols, picks, n, "int32")

def conv_kernel(n):
  """The kernel for conv: the SiLU of the sum over n taps of the table's picked rows times the tap, one result tile a
  unit, in fp32.

  Arguments: the first unit and the number of units, then four words for each tensor (DRAM address, batch tile
  stride, row tile stride, tile offset): the result, the table, the picks (an int32 row for each tap) and the taps (a
  row for each), then the result's tile columns and rows, then last's four words, the table's rows and last's: a pick
  past the table's rows p reads last's row (p - rows) as conv takes it.

  For each unit, BRISC reads its picks once a tile row, then the taps' tiles (a 1-row tensor repeats its row down its
  tiles), and each tap's picked rows: from the table's tiles that they reach, or last's rows alone, through a cache of
  4, by NoC copies within L1, a run of rows in order at a time, since a row's run of 16 values in a face is a whole
  64-byte block. The math thread sums the products on the SFPU. Frames take turns, so BRISC reads one unit while math
  works on the last. The result's padding rows stay 0, as SiLU(0) is."""
  OUT, F0, R2, R3 = 1, 3, 2 * DST_HALF, 3 * DST_HALF  # output slots for each Dst half, then a frame of 2n tiles each
  IDS, CACHE = F0 + 4 * n, F0 + 5 * n  # the picks' tiles, then the table's
  if CACHE + 4 > UNPACKABLE: raise ValueError("conv_kernel: too many taps for L1")
  frame = f"{F0} + (u & 1) * {2 * n}"
  units = lambda body: f"for (unsigned u = 0; u < a[1]; u++) {{ unsigned t = a[0] + u, i = t / a[18], j = t % a[18]; {body} }}"
  held = "unsigned held = ~0u, tag[4], next = 0;  // the picks' tile row, and the cache's, for this launch\n"
  brisc = held + units(f"""unsigned F = {frame}, rows = a[19] - i * 32 < 32 ? a[19] - i * 32 : 32;
    reached(frame_done[u & 1]);
    if (i != held) {{ for (unsigned k = 0; k < {n}; k++) get(a[10], {tile_of(10, 'k', 0, 'i')}, {IDS} + k); held = i; }}
    for (unsigned k = 0; k < {n}; k++) {{
      get(a[14], {tile_of(14, 'k', 0, 'j')}, F + {n} + k);
      if (rows < 32) copy({ZEROS:#x}, SLOT(F + k), 4096);  // full tiles overwrite every row; partial ones keep zeros
    }}
    noc_wait();
    for (unsigned c = 0; c < 4; c++) tag[c] = ~0u;  // the cache holds tiles of this unit's column
    for (unsigned k = 0, run; k < {n}; k++) for (unsigned r = 0; r < rows; r += run) {{
      volatile unsigned *picks = (volatile unsigned *)SLOT({IDS} + k);
      unsigned p = picks[at(0, r)], L = p >= a[24], b = L ? (p - a[24]) / a[25] : 0;
      // the row, in the table or last, and what the cache holds it in: the table's tile, or last's row alone
      unsigned w = L ? (p - a[24]) % a[25] : p, q = (L ? w : w / 32) | b << 16 | L << 31, c = 0;
      while (c < 4 && tag[c] != q) c++;
      if (c == 4) {{  // a miss takes the next place, once the copies from it are done
        c = next++ % 4, tag[c] = q;
        noc_wait();
        if (L) for (unsigned f = 0, t = {tile_of(20, 'b', 'w / 32', 'j')}, o; f < 2; f++)
          o = at(w % 32, 16 * f) * 4, noc_get(BANK(t), AT(a[20], t, 4096) + o, SLOT({CACHE} + c) + o, 64);
        else get(a[6], {tile_of(6, 0, 'w / 32', 'j')}, {CACHE} + c);
        noc_wait();
      }}
      // a run of the table's rows in order goes in one copy a face, within a face's 16 rows on both sides
      for (run = 1; !L && r + run < rows && (r + run) % 16 && (w + run) % 16 && p + run < a[24] &&
                    picks[at(0, r + run)] == p + run;) run++;
      for (unsigned f = 0; f < 2; f++)
        copy(SLOT({CACHE} + c) + at(w % 32, 16 * f) * 4, SLOT(F + k) + at(r, 16 * f) * 4, 64 * run);
    }}
    noc_wait(); ready(); frame_done[u & 1] = ++events;""")
  free = [ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_SFPU)]
  fills = lambda *pairs: tt(free + [w for slot, row in pairs for w in fill(slot, row)] + FILLED)
  lib = lambda name: f"{tt(aim(R2))} p_{name}(); {tt(aim(0))}"  # an SFPU library program, on the tile at R2
  silu = f"p_neg(h); {lib('fexp')} p_inc(h); {lib('frecip')} p_mul(h);"  # x / (1 + e^-x), with e^-x at R2
  math = units(f"unsigned F = {frame}; fetched(); {tt([DST_FREE])} {fills(('F', 'h'), (f'F + {n}', R2))} p_mul(h); " +
               "".join(f"{fills((f'F + {k}', R2), (f'F + {n + k}', R3))} p_fma(h); " for k in range(1, n)) +
               f"{silu} {tt(MATH_END)} h ^= {DST_HALF}; {tt(REPORT)}")
  packer = units(f"output(h / {DST_HALF}, {OUT} + h / {DST_HALF}, {tile_of(2, 0, 'i', 'j')}); "
                 f"{tt(pack('h', f'{OUT} + h / {DST_HALF}', SEM_PACKED))} h ^= {DST_HALF};")
  programs = {"mul": lambda r: [sfp_load(0, r), sfp_load(1, R2), mul(2, 0, 1), sfp_store(2, r)],
              "fma": lambda r: [sfp_load(0, r), sfp_load(1, R2), sfp_load(2, R3), mad(3, 1, 2, 0), sfp_store(3, r)],
              "neg": lambda r: [sfp_load(0, r), mul(1, 0, MINUS_ONE), sfp_store(1, R2)],  # -r into R2
              "inc": lambda r: [sfp_load(0, R2), mad(1, 0, ONE, ONE), sfp_store(1, R2)]}  # 1 + R2
  library = "".join(f"__attribute__((noinline)) static void p_{n}(void) {{ {tt(sfpu([n], {}))} }}\n" for n in ("fexp", "frecip"))
  cfg = kernel_config("fp32", "fp32")
  return compile_kernel(brisc_program("a[1]", brisc, "fp32", "fp32", COPIES),
                        unpack_program(units("fetched();"), cfg), math_program({}, math, True, helpers(programs) + library),
                        pack_program(packer))

def conv(table, picks, taps, last=None):
  """SiLU of the sum over taps t of table[picks[t]] times taps[t]: Gated DeltaNet's conv, for tokens whose inputs are
  rows of table in any order, such as a prompt's in turn or a tree's along each path (conv_kernel, one launch). table
  is a matrix of fp32 rows, picks an int32 row of indices for each tap, and taps a row of fp32 weights for each. With
  last, the conv state as deltanet keeps it (conv - 1, sequences, width), a pick p past table's rows reads row
  p - table.rows of last's, sequence b's j-th last input at j sequences + b, where it lies. Unchecked indices, as
  Tensor.gather's."""
  if isinstance(table, np.ndarray):
    table = table.reshape(-1, table.shape[-1])
    table = table if last is None else np.concatenate([table, last.reshape(-1, table.shape[-1])])
    y = sum(table[np.ravel(p)] * w for p, w in zip(picks, taps))[None]
    return y / (1 + np.exp(-y))
  m, width = picks.cols, table.cols
  if not (table.batch == 1 and table.dtype == taps.dtype == "fp32" and picks.dtype == "int32" and picks.rows == 1 and
          taps.shape == (picks.batch, 1, width) and (last is None or (last.cols, last.dtype) == (width, "fp32"))):
    raise ValueError("conv needs an fp32 table, an int32 row of picks and an fp32 row of taps for each tap")
  out = Tensor(table.rt, m, width, dtype="fp32")
  return out.run(("conv", picks.batch), lambda: conv_kernel(picks.batch), out, table, picks, taps, -(-width // 32),
                 m, *((table, 0xFFFFFFFF, 1) if last is None else (last, table.rows, last.rows)), units=out.tiles, name="conv")

def conv_state_kernel():
  """The kernel for conv_state, BRISC's alone, a unit a tile column. Arguments: the first unit and the number of
  units, then four words each for last and the table (DRAM address, batch tile stride, row tile stride, tile offset)
  and the picks, then the table's rows, last's rows and its batch. A unit reads each new row's two runs of 16 into
  L1, from the table or from last as it was, before it writes any, then writes the runs into their rows of last, as
  delta_kernel writes the conv state."""
  get = "noc_get(BANK(from), AT(a[L ? 2 : 6], from, 4096) + at(r, 16 * f) * 4, RUNS + (2 * i + f) * 64, 64)"
  put = "noc_put(RUNS + (2 * i + f) * 64, BANK(to), AT(a[2], to, 4096) + at(i % a[15], 16 * f) * 4, 64)"
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define RUNS {DATA + 4096:#x}
{BANKS}
void kmain(volatile unsigned *a) {{
  for (unsigned o = 0; o < 2048; o += 1024) noc_get(BANK(a[13]), AT(a[10], a[13], 4096) + o, IDS + o, 64);
  noc_wait();
  for (unsigned u = 0, n = a[15] * a[16]; u < a[1]; u++) {{
    unsigned j = a[0] + u;
    for (unsigned i = 0; i < n; i++) {{  // input i / sequences of sequence i % sequences, from the table or last
      unsigned p = ((volatile unsigned *)IDS)[at(0, i)], L = p >= a[14], q = p - a[14], r = L ? q % a[15] : p % 32;
      unsigned from = L ? {tile_of(2, 'q / a[15]', 0, 'j')} : {tile_of(6, 0, 'p / 32', 'j')};
      for (unsigned f = 0; f < 2; f++) {get};
    }}
    noc_wait();
    for (unsigned i = 0, to; i < n; i++) for (unsigned f = 0; f < 2; f++) to = {tile_of(2, 'i / a[15]', 0, 'j')}, {put};
    noc_wait();
  }}
}}""")

def conv_state(last, table, picks):
  """Write into the conv state last (conv - 1, sequences, width, as deltanet keeps it), in place, the rows that picks
  names, as conv's do: sequence b's input j is pick j sequences + b, a row of table or, past its rows, of last as it
  was. picks is an int32 row, its indices unchecked, as conv's. On the chip, conv_state_kernel's one launch: commit's
  conv state after a path, from verify's rows and the state before it."""
  if isinstance(last, np.ndarray):
    rows = np.concatenate([table.reshape(-1, table.shape[-1]), last.reshape(-1, last.shape[-1])])[np.ravel(picks)]
    last[...] = rows.reshape(last.shape)
    return last
  if not (table.batch == 1 and table.dtype == last.dtype == "fp32" and table.cols == last.cols and
          picks.dtype == "int32" and picks.shape[-2:] == (1, last.batch * last.rows) and picks.cols <= 32):
    raise ValueError("conv_state needs fp32 rows of the state's width, and an int32 row of 32 picks or fewer")
  return last.run(("conv_state",), conv_state_kernel, last, table, picks, table.rows, last.rows, last.batch,
                  units=-(-last.cols // 32), name="conv_state")

R2, R3 = 2 * DST_HALF, 3 * DST_HALF  # where an op fills tiles for the SFPU beside its result, as Dst rows for half 0

class Tiles:
  """A kernel of tile ops for one unit at a time, all in fp32 (rule_kernel, carry_kernel). Every product runs on the
  FPU, which reads fp32 tiles as TF32 (10 bits of mantissa, as the GPU engine's tree kernel), and everything else on
  the SFPU. The unpacker truncates to TF32, a bias that sums over K, so every tile the FPU reads is rounded to nearest
  first: op rounds what it keeps, unless it's exact, for the SFPU alone. The math thread unpacks everything: into Dst
  for the SFPU, into SrcA and SrcB for the FPU. Each op makes one tile in a Dst half, which pack writes into L1 for the
  ops after it, or for BRISC to write out to a tile of the result.

  A unit's tiles are L1 slots from take. What the packer writes comes first, after slot 0, then the output slots
  (outs), all within bh.PACK_REACH of slot 0; the inputs follow. BRISC fetches a unit's inputs once math has reported
  the last unit done, and writes results out as they're packed."""
  programs = {"mul": lambda r: [sfp_load(0, r), sfp_load(1, R2), mul(2, 0, 1), sfp_store(2, r)],
              "rsub": lambda r: [sfp_load(0, r), sfp_load(1, R2), mad(2, 0, ONE, 1, neg=1), sfp_store(2, r)],  # R2 - r
              "neg": lambda r: [sfp_load(0, r), mul(1, 0, MINUS_ONE), sfp_store(1, r)],
              "fma": lambda r: [sfp_load(0, r), sfp_load(1, R2), sfp_load(2, R3), mad(3, 1, 2, 0), sfp_store(3, r)],
              "round": lambda r: [sfp_load(0, r), *SFPU["ftoh"][1][1:-1], sfp_store(0, r, mod0=RAW)],  # to TF32's 10 bits
              # what rounding to TF32 leaves: x - round(x), exact in fp32, and min(x, 0)
              "lo": lambda r: [sfp_load(0, r), *SFPU["ftoh"][1][1:-1], sfp_load(5, r), mad(6, 0, ONE, 5, neg=1), sfp_store(6, r)],
              "min0": lambda r: [sfp_load(0, r), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=0, vc=1, mod1=1),
                                 sfp_store(0, r)],
              "square": lambda r: [sfp_load(0, r), mul(1, 0, 0), sfp_store(1, r)],
              "eps": lambda r: [sfp_load(0, r), *sfp_const(5, 1e-6), mad(1, 0, ONE, 5), sfp_store(1, r)],  # + 1e-6
              # for the gates: r + R2, 1 + r, -|r| and r + max(R2, 0)
              "add": lambda r: [sfp_load(0, r), sfp_load(1, R2), mad(2, 0, ONE, 1), sfp_store(2, r)],
              "inc": lambda r: [sfp_load(0, r), mad(1, 0, ONE, ONE), sfp_store(1, r)],
              "nabs": lambda r: [sfp_load(0, r), ins("SFPABS", vd=1, vc=0, mod1=1), mul(2, 1, MINUS_ONE),
                                 sfp_store(2, r)],
              "addmax0": lambda r: [sfp_load(0, R2), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1),
                                    sfp_load(2, r), mad(3, 2, ONE, 0), sfp_store(3, r)]}

  library = {"fexp": sfpu(["fexp"], {}), "sqrt": sfpu(["sqrt"], {}), "frecip": sfpu(["frecip"], {}),
             "log": sfpu(["log"], {}), "sum2": ACROSS[FOLDS["sum2"]]("sum2"), "col0": mask_columns("sum", 1)}
  # on the tile at h: fp32 e^x, sqrt, 1 / x and log, each row's sum filling it, and column 0 alone

  def __init__(self):
    self.used, self.ops, self.made, self.drained, self.keeps, self.replays = 1, [], {}, 0, {}, allocate({"mm", "mmT"})
    self.programs, self.called = dict(Tiles.programs), set()  # a kernel's own programs, and the library's it calls

  def take(self, n, cols=None):
    """n slots, from the first unused: the first, or with cols, tile (i, j) of rows of cols tiles."""
    self.used += n
    base = self.used - n
    return base if cols is None else (lambda i, j: base + i * cols + j)

  def outs(self):  # an output slot for each Dst half, after what the packer writes
    self.OUT = self.take(2)
    return self.OUT

  def op(self, slot, terms=(), start=None, posts=(), home=None, exact=False):
    """One tile into slot: the sum of the products in terms, as (A's tile, B's tile, B transposed), or start's tile,
    then posts, (program, tiles) in turn, each with its tiles filled beside the result. With home, it goes out to that
    tile of the result, and stays in L1 if slot isn't OUT. A kept tile is rounded to TF32 for the FPU, unless it's
    exact, for the SFPU alone. A program may be the library's (Tiles.library)."""
    srca = lambda on: [ins("SETC16", setc16_reg=bh.THREAD_CFG[r], setc16_value=v) for r, v in  # unpacker 0 into SrcA, or Dst
                       (("UNPACK_MISC", 1 - on), ("SRCA_SET", bh.SRCA_SET_BY_ADDR * on))]
    formats = lambda f: reconfigure(SEC0_out_format=f, SEC1_out_format=f)
    free = [ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_SFPU)]  # a fill waits for the SFPU to finish
    read = [a for a, _, _ in terms] + [b for _, b, _ in terms] + [t for _, ts in posts for t in ts] + ([start] if start else [])
    # Wait for what pack has put in L1 before reading it, and before SEM_L1 would hold more than 15
    waiting = sum(1 for j in self.made.values() if j >= self.drained)
    words = ["drain();"] if any(self.made.get(t, -1) >= self.drained for t in read) or waiting >= 15 else []
    if words: self.drained = len(self.ops)
    if terms:
      mm, was = math_begin(True) + srca(1) + formats(bh.TF32), False
      for a, b, T in terms:  # a transposed B comes as it lies: unpacker 0 turns its faces, and the walk picks them
        if T != was: mm += haloize(T) + walk(T)
        mm, was = mm + unpack(b, a) + matmul(self.replays, 4, T), T
      if was: mm += haloize(0) + walk(False)
      words.append(tt(mm + formats(bh.FP32) + srca(0) + aim(0)))
    else: words.append(tt([DST_FREE] + free + fill(start, "h") + FILLED))
    kept = home is None or slot != self.OUT
    for p, ts in [*posts, *[("round", ())] * (kept and not exact)]:  # an FPU operand: TF32 to nearest, unbiased
      if p in self.library: self.called.add(p)
      run = f"{tt(aim('h'))} p_{p}(); {tt(aim(0))}" if p in self.library else f"p_{p}(h);"
      words.append((f"{tt(free + [w for t, r in zip(ts, (R2, R3)) for w in fill(t, r)] + FILLED)} " if ts else "") + run)
    words.append(f"{tt(MATH_END)} h ^= {DST_HALF};")
    if kept: words.append("posted++;"); self.made[slot] = len(self.ops)
    if home is None: pack_ = f"{tt(pack('h', slot, SEM_L1))}"
    else:
      to, keep = ((f"{self.OUT} + h / {DST_HALF}", f"h / {DST_HALF}") if slot == self.OUT else
                  (slot, self.keeps.setdefault(slot, 2 + len(self.keeps))))
      pack_ = f"output({keep}, {to}, {tile_of(2, 'e', home[0], home[1])}); {tt(pack('h', to, SEM_PACKED, *[SEM_L1] * kept))}"
    self.ops.append((" ".join(words), f"{pack_} h ^= {DST_HALF};"))

  def norm(self, slot, row, kt, scale=1.0):
    """slot's tile filled with scale / |r| for each row r of the kt tiles from row on, as qwen.unit: the squares summed
    on the SFPU in fp32, then along each row. A kernel's norms other than 1 / |r| share one scale."""
    if scale != 1:  # one scale a kernel
      self.programs["scale"] = lambda r: [sfp_load(0, r), *sfp_const(5, scale), mul(1, 0, 5), sfp_store(1, r)]
    self.op(slot, start=row, posts=[("square", ()), *[("fma", (row + c, row + c)) for c in range(1, kt)], ("sum2", ()),
                                    ("eps", ()), ("sqrt", ()), ("frecip", ()), *[("scale", ())] * (scale != 1)], exact=True)

  def parts(self, at, i, **made):  # a tile in two TF32 parts, rounded and what that leaves, for the FPU: 22 bits of it
    for p in range(2): self.op(at(i, p), **made, posts=[("lo", ())] * p)

  @staticmethod
  def get(i, base, rows, cols, b="e"):  # BRISC's fetch of rows x cols tiles of the tensor at word i, from slot base on
    return (f"for (unsigned r = 0; r < {rows}; r++) for (unsigned c = 0; c < {cols}; c++) "
            f"get(a[{i}], {tile_of(i, b, 'r', 'c')}, {base} + r * {cols} + c);")

  def kernel(self, name, fetch, total):
    """The kernel of the ops, with fetch, BRISC's fetch of a unit's inputs, and total, the tiles a unit writes out."""
    if (self.OUT + 2) * 4096 > bh.PACK_REACH or self.used > UNPACKABLE or slot_base("fp32") + self.used * 4096 > OUTS:
      raise ValueError(f"{name}: a unit's tiles don't fit in L1")
    units = lambda body: f"for (unsigned u = 0; u < a[1]; u++) {{ unsigned e = a[0] + u; {body} }}"
    brisc = units(f"reached(frame_done[0]); {fetch} ready(); frame_done[0] = ++events;")
    cfg = kernel_config("fp32", "fp32")
    library = "".join(f"__attribute__((noinline)) static void p_{n}(void) {{ {tt(self.library[n])} }}\n" for n in sorted(self.called))
    return compile_kernel(brisc_program(f"a[1] * {total}", brisc, "fp32", "fp32"), unpack_program(units("fetched();"), cfg),
                          math_program(self.replays, units("fetched(); " + " ".join(m for m, _ in self.ops) + " " + tt(REPORT)),
                                       True, helpers(self.programs, {"round": SFPU["ftoh"][0], "lo": SFPU["ftoh"][0]}) + library,
                                       wide=True),
                          pack_program(units(" ".join(c for _, c in self.ops)), 2 + len(self.keeps)))

def rule_kernel(st, kt, vt, rep, state, kd=None, gated=None):
  """The kernel for rule: a piece's gated delta rule after the decays, one value head a unit, in Tiles' ops. The
  inputs come rounded to TF32 by rule, except the state, which goes on in fp32 and is rounded here for the FPU.

  Arguments: the first unit and the number of units, then four words for each tensor (DRAM address, batch tile
  stride, row tile stride, tile offset), in this order:
  - out: for each value head, its outputs, then v_new, then with state the state after the piece
  - q and k, for each key head; a unit reads those of its value head's (e / rep)
  - kb = beta k, kbe = beta k e^cum, qe = q e^cum and vb = beta v, for each value head, then decay: the decay from
    token j to token i where i follows j (or is it), else 0. With kd, v, beta as tiles of each row's value, g and
    seen (corrected's) in their place, or with gated, v, then b, a, dt_bias and decay as deltanet has them (rule's
    raw) in place of beta and g, seen, and z and the norm's weight
  - S, the state before the piece
  - consts: the identity, the pairs masks of spans 1, 2, 4, 8 and 16 (llama.pairs), ones in row 0, ones in column 0
  - with state, e^G, where G is cum at the last token, in every place of a tile, and kl = (k e^(G - cum))^T

  With kd, the width of a key, for a piece of one tile without state, whose made tiles fit the packer's reach, q and k
  come as the conv gives them, and the kernel takes them to length 1 (q then times kd^-1/2, as qwen.unit), and makes
  cum, which goes out after v_new, its tiles filled, the decays, and the rounded q, k, kb, kbe, qe and vb, each
  rounded once. cum = seen g, and cum's column filling its tiles, or turned for cum_i - cum_j, come from the FPU, from two
  TF32 parts of each value: 22 bits of it, where the tensor ops' fp32 matmul has bf16's two parts, 16. With gated, it
  makes beta = sigmoid(b) and g = decay softplus(a + dt_bias) too, for one sequence: BRISC copies the unit's column
  of each into column 0, and the SFPU spreads each row's value along its row. a + dt_bias and g go in slots of the
  inverse's, which it makes later. Then each output row is RMS-normed with eps, times the weight and gated by silu(z),
  as deltanet's last step, the outputs kept in vb's slots and their norm in L's, which are spent by then. gated is
  deltanet's eps and the width of a value, for the norm.

  The ops follow corrected and chunk: L = (kb k^T) decay, then the inverse of I + L in blocks, w = inv kbe (kept
  negated), v_new = inv vb - w S, the outputs qe S + ((q k^T) decay) v_new, and the state e^G S + kl v_new. L's diagonal
  is never read, since every pairs mask lies below it. st, kt and vt are the tiles of a piece, a key and a value."""
  t, scaled = Tiles(), kd is not None
  L, INV, LP, X = (t.take(st * st, st) for _ in range(4))
  NW, VN, QK = t.take(st * kt, kt), t.take(st * vt, vt), t.take(st * st, st)
  SR = t.take(kt * vt, vt)  # the state rounded, for the FPU
  scaled_rows = lambda: ([t.take(st * kt, kt) for _ in range(5)], t.take(st * vt, vt))
  if scaled:  # made here, so within the packer's reach: the scaled rows, g and cum in two TF32 parts, cum filling
    (Q, K, KB, KBE, QE), VB = scaled_rows()  # its tiles, e^cum and the decays
    GP, CP, CU, EE, DC = t.take(2 * st, 2), t.take(2 * st, 2), t.take(st), t.take(st), t.take(st * st, st)
    if gated: BE, TM, GI = t.take(st), X(0, 0), LP(0, 0)
  OUT = t.outs()
  if scaled:
    QR, KR, VR = t.take(st * kt, kt), t.take(st * kt, kt), t.take(st * vt, vt)
    if gated: BR = t.take(4)  # the unit's tile of b, a, dt_bias and decay, its column first
    else: BE, GI = t.take(st), t.take(st)
    SN = t.take(st * st, st)
    if gated: Z, WT = t.take(vt), t.take(vt)  # the unit's tiles of z, and the norm's weight
  else:
    (Q, K, KB, KBE, QE), VB = scaled_rows()
    DC = t.take(st * st, st)
  SS, CN = t.take(kt * vt, vt), t.take(8)
  E0, E0T = CN + 6, CN + 7  # ones in row 0, ones in column 0
  EG, KL = (t.take(1), t.take(kt * st, st)) if state else (None, None)
  op, lower = t.op, [(i, j) for i in range(st) for j in range(i + 1)]
  spread = lambda i: [(CP(i, p), E0, False) for p in range(2)]  # cum's column filling its tiles
  row = [("col0", ()), ("sum2", ())]  # each row's value in column 0, filling its row
  if gated:  # beta = sigmoid(b), and g = decay softplus(a + dt_bias) = decay (log(1 + e^-|x|) + max(x, 0))
    op(BE, start=BR, posts=[*row, ("neg", ()), ("fexp", ()), ("inc", ()), ("frecip", ())], exact=True)
    op(TM, start=BR + 1, posts=[("add", (BR + 2,)), *row], exact=True)
    softplus = [("nabs", ()), ("fexp", ()), ("inc", ()), ("log", ()), ("addmax0", (TM,))]
    op(GI, start=TM, posts=[*softplus, ("mul", (BR + 3,)), *row], exact=True)
  if scaled:  # cum = seen g, then filling its tiles (its column times ones in row 0), e^cum, and the decays:
    for i in range(st):  # exp(min(cum_i - cum_j, 0)) where i sees j, cum_j's tile turned by ones in column 0
      t.parts(GP, i, start=GI + i)
      t.parts(CP, i, terms=[(SN(i, j), GP(j, p), False) for j in range(i + 1) for p in range(2)])
      op(CU + i, spread(i), exact=True), op(OUT, spread(i), home=(2 * st + i, 0))
      op(EE + i, start=CU + i, posts=[("fexp", ())], exact=True)
    for i, j in lower:
      op(DC(i, j), [(E0T, CP(j, p), True) for p in range(2)],
         posts=[("rsub", (CU + i,)), ("min0", ()), ("fexp", ()), ("mul", (SN(i, j),))], exact=True)
    NQ, NK = (lambda i: GP(i, 0)), (lambda i: GP(i, 1))  # 1 / |q| times kd^-1/2, and 1 / |k|, in g's parts' slots
    for i in range(st):  # then the scaled rows, each rounded once (op rounds what it keeps)
      t.norm(NQ(i), QR(i, 0), kt, kd ** -0.5), t.norm(NK(i), KR(i, 0), kt)
      for c in range(kt):
        q, k = ("mul", (NQ(i),)), ("mul", (NK(i),))
        op(Q(i, c), start=QR(i, c), posts=[q]), op(K(i, c), start=KR(i, c), posts=[k])
        op(KB(i, c), start=KR(i, c), posts=[k, ("mul", (BE + i,))])
        op(KBE(i, c), start=KR(i, c), posts=[k, ("mul", (BE + i,)), ("mul", (EE + i,))])
        op(QE(i, c), start=QR(i, c), posts=[q, ("mul", (EE + i,))])
      for j in range(vt): op(VB(i, j), start=VR(i, j), posts=[("mul", (BE + i,))])
  for c in range(kt):
    for j in range(vt): op(SR(c, j), start=SS(c, j))  # rounded, as rule rounds the other inputs
  for i, j in lower: op(L(i, j), [(KB(i, c), K(j, c), True) for c in range(kt)], posts=[("mul", (DC(i, j),))])
  for i in range(st): op(INV(i, i), start=L(i, i), posts=[("mul", (CN + 1,)), ("rsub", (CN,))])  # I - L, a token at a time
  for n, span in enumerate((2, 4, 8, 16)):  # join blocks within each tile: inv -= (inv (L pairs)) inv
    for i in range(st):
      op(LP(i, i), start=L(i, i), posts=[("mul", (CN + 2 + n,))])
      op(X(i, i), [(INV(i, i), LP(i, i), False)])
      op(INV(i, i), [(X(i, i), INV(i, i), False)], posts=[("rsub", (INV(i, i),))])
  span = 1
  while span < st:  # then blocks of tiles: below the diagonal of each pair of blocks, -B^-1 C A^-1
    for b0 in range(0, st, 2 * span):
      second, first = range(b0 + span, b0 + 2 * span), range(b0, b0 + span)
      for i in second:
        for m in first: op(X(i, m), [(INV(i, l), L(l, m), False) for l in second if l <= i])
        for j in first: op(INV(i, j), [(X(i, m), INV(m, j), False) for m in first if m >= j], posts=[("neg", ())])
    span *= 2
  for i in range(st):
    for c in range(kt): op(NW(i, c), [(INV(i, l), KBE(l, c), False) for l in range(i + 1)], posts=[("neg", ())])
  for i in range(st):
    for j in range(vt):
      op(VN(i, j), [(INV(i, l), VB(l, j), False) for l in range(i + 1)] + [(NW(i, c), SR(c, j), False) for c in range(kt)],
         home=(st + i, j))
  for i, j in lower: op(QK(i, j), [(Q(i, c), K(j, c), True) for c in range(kt)], posts=[("mul", (DC(i, j),))])
  for i in range(st):
    for j in range(vt):
      terms = [(QE(i, c), SR(c, j), False) for c in range(kt)] + [(QK(i, l), VN(l, j), False) for l in range(i + 1)]
      if gated: op(VB(i, j), terms, exact=True)
      else: op(OUT, terms, home=(i, j))
    if gated:  # o rsqrt(mean(o^2) + eps) weight silu(z), with o in vb's slots, its norm in L's and silu(z) in LP's
      eps, vd = gated
      t.programs["rms"] = lambda r: [sfp_load(0, r), *sfp_const(5, 1 / vd), *sfp_const(6, eps), mad(1, 0, 5, 6),
                                     sfp_store(1, r)]  # the mean of the squares, plus eps
      op(L(i, i), start=VB(i, 0), posts=[("square", ()), *[("fma", (VB(i, j), VB(i, j))) for j in range(1, vt)],
                                         ("sum2", ()), ("rms", ()), ("sqrt", ()), ("frecip", ())], exact=True)
      for j in range(vt):
        op(LP(i, i), start=Z + j, posts=[("neg", ()), ("fexp", ()), ("inc", ()), ("frecip", ()), ("mul", (Z + j,))],
           exact=True)
        op(OUT, start=VB(i, j), posts=[("mul", (L(i, i),)), ("mul", (WT + j,)), ("mul", (LP(i, i),))], home=(i, j))
  if state:
    for c in range(kt):
      for j in range(vt):  # kl v_new + e^G S, with S as it came
        op(OUT, [(KL(c, l), VN(l, j), False) for l in range(st)], posts=[("fma", (SS(c, j), EG))], home=(2 * st + c, j))
  gates = ["b", "a", "dt", "decay"] if gated else ["beta", "g"]
  pieces = ["v", *gates, "seen", *(["z", "weight"] if gated else [])] if scaled else ["kb", "kbe", "qe", "vb", "decay"]
  names = ["q", "k", *pieces, "S", "consts",
           *(["eg", "kl"] if state else [])]
  arg, get = {name: 6 + 4 * i for i, name in enumerate(names)}, t.get  # each input's four words, after the result's
  rows = ((QR, KR, VR) if scaled else (Q, K, VB))
  raw = ""
  if gated:  # the gates' tiles, each with the unit's column copied into column 0 (a 1-row tensor repeats its row)
    raw = "".join(f"get(a[{arg[n]}], {tile_of(arg[n], 0, 0, 'e / 32')}, {BR + j});" for j, n in enumerate(gates))
    raw += (f"noc_wait(); for (unsigned s = 0; s < 4; s++) {{ volatile unsigned *x = (volatile unsigned *)"
            f"SLOT({BR} + s); for (unsigned r = 0; r < 32; r++) x[at(r, 0)] = x[at(r, e % 32)]; }}")
    z, w = arg["z"], arg["weight"]
    raw += (f"for (unsigned c = 0; c < {vt}; c++) get(a[{z}], {tile_of(z, 0, 0, f'e * {vt} + c')}, {Z} + c), "
            f"get(a[{w}], {tile_of(w, 0, 0, 'c')}, {WT} + c);")
  fetch = (get(arg["q"], rows[0](0, 0), st, kt, f"e / {rep}") + get(arg["k"], rows[1](0, 0), st, kt, f"e / {rep}") +
           (get(arg["v"], VR(0, 0), st, vt) + (raw or get(arg["beta"], BE, st, 1) + get(arg["g"], GI, st, 1))
            + get(arg["seen"], SN(0, 0), st, st) if scaled else
            "".join(get(arg[n], m(0, 0), st, kt) for n, m in (("kb", KB), ("kbe", KBE), ("qe", QE))) +
            get(arg["vb"], VB(0, 0), st, vt) + get(arg["decay"], DC(0, 0), st, st)) + get(arg["S"], SS(0, 0), kt, vt) +
           f"for (unsigned t = 0; t < 8; t++) get(a[{arg['consts']}], {tile_of(arg['consts'], 't', 0, 0)}, {CN} + t);" +
           (get(arg["eg"], EG, 1, 1) + get(arg["kl"], KL(0, 0), kt, st) if state else ""))
  return t.kernel("rule_kernel", fetch, 2 * st * vt + kt * vt * state + st * scaled)

def unit(t, scale=1.0):  # t's rows to length scale, as transformers' l2norm of Gated DeltaNet's q and k
  f = llama.fns(t)
  return t * f.rsqrt(f.mean(t * t) * t.shape[-1] + 1e-6) * scale

def rule(q, k, v, beta, g, S, seen=None, state=False, normed=False, raw=None):
  """A piece's gated delta rule, from state S: each token's output and its v_new, cum (corrected's), and with state,
  the state after the piece. q and k have a matrix for each key head, which serves the value heads that follow it, as
  the conv gives them: rule takes them to length 1, q then times kd^-1/2 (unit), as transformers'
  use_qk_l2norm_in_kernel, unless they come normed. seen is as corrected takes it. On the chip it's rule_kernel's one
  launch, for a piece of one tile without state, from q and k as the conv gives them; with more, cum, the decays and
  the scaled rows come from tensor ops in fp32 before it. With raw, deltanet's for one sequence (b and a as x's
  columns, dt_bias, decay, z as x's columns, the norm's weight and eps), the outputs come RMS-normed and gated by z,
  deltanet's last step, which that launch does itself, with beta and g, so beta and g aren't read."""
  f, s, rep = llama.fns(q), q.shape[-2], v.shape[0] // q.shape[0]
  if raw and raw[4].shape[-1] != v.shape[0] * v.shape[-1]: raise ValueError("rule's raw is one sequence's")
  # the norm as tensor ops, where the kernel doesn't make it
  norm = lambda o: o * f.rsqrt(f.mean(o * o) + raw[6]) * raw[5] * f.silu(f.split(raw[4], v.shape[0])) if raw else o
  # on the chip for a piece of one tile, the kernel takes q and k to length 1 and makes cum, the decays and the rows
  scaled = isinstance(q, Tensor) and s <= 32 and not (state or normed)
  if not (normed or scaled): q, k = unit(q, q.shape[-1] ** -0.5), unit(k)
  if isinstance(q, np.ndarray):
    q, k = f.repeat(q, rep), f.repeat(k, rep)
    cum, decay, u, w = corrected(k, v, beta, g, seen)
    vnew = u - w @ S
    o = norm((q * np.exp(cum)) @ S + ((q @ f.t(k)) * decay) @ vnew)
    if not state: return o, vnew, cum
    G = cum[:, s - 1:s, :]
    return o, vnew, cum, S * np.exp(G) + f.t(k * np.exp(G - cum)) @ vnew
  (h, kd, vd), (st, kt, vt) = (S.batch, S.rows, S.cols), (-(-n // 32) for n in (s, S.rows, S.cols))
  if q.batch * rep != h or k.shape != q.shape: raise ValueError("rule needs q and k for the key heads of the value heads")
  # The FPU reads its operands as TF32, which the unpacker truncates, a bias that sums over K, so they're rounded first,
  # each once
  near = lambda t: t.rounded("fp16")
  qr, kr, seen = f.repeat(q, rep), f.repeat(k, rep), f.tri(s, True) if seen is None else seen
  if not scaled:
    cum = f.now(f.bf16(seen) @ g)  # seen is 0 or 1, which bf16 holds exactly
    decay = f.exp(f.minimum(cum - f.t(cum), 0)) * seen
  extra = (f.exp(G := cum[:, s - 1:s, :]) * ones(q.rt), near(f.t(kr * f.exp(G - cum)))) if state else ()
  out = Tensor(q.rt, 64 * st + 32 * (kt * state + scaled), vd, batch=h, dtype="fp32")  # its padding stays 0, as the inputs' does
  gated = (raw[6], vd) if scaled and raw else None
  if scaled: rows = (q, k, v, *(raw[:4] if gated else (beta * ones_row(q.rt), g)), seen, *(raw[4:6] if gated else ()))
  else:
    e, beta = f.now(f.exp(cum)), f.now(beta)  # columns, made once: the scaled rows would each work them out again
    rows = (*map(near, (q, k, kb := kr * beta, kb * e, qr * e, v * beta)), decay)
  key = ("rule", st, kt, vt, rep, state, q.shape[-1] if scaled else None, gated)
  out.run(key, lambda: rule_kernel(*key[1:]), out, *rows, S, constants(q.rt), *extra, units=h, name="rule")
  o, vnew = out[:, :s, :] if gated else norm(out[:, :s, :]), out[:, 32 * st:32 * st + s, :]
  if scaled: cum = out[:, 64 * st:64 * st + s, :1]  # a copy of its tiles' first column
  return (o, vnew, cum, out[:, 64 * st:64 * st + kd, :]) if state else (o, vnew, cum)

def carry_kernel(st, kt, vt, rep, per):
  """The kernel for carry, one value head a unit, in Tiles' ops: w = on e^min(G - cum, 0) filling its tiles, where G
  is cum at end (end's row filled down, times cum's column filled, from two TF32 parts), then S = e^G S + (k w)^T
  v_new, k w turned by the identity, with S as it came. st, kt and vt are as rule_kernel's.

  Arguments: the first unit and the number of units, then four words for each tensor (as rule_kernel's): S, which
  the kernel writes over; k, for each key head (a unit reads its value head's, e / rep); v_new and cum, for each
  value head; end and on, for each sequence of per value heads; consts (constants). k comes as the conv gives it, and
  the kernel takes it to length 1 (Tiles.norm)."""
  t = Tiles()
  KW, KT, VR, CP, CF = t.take(st * kt, kt), t.take(kt * st, st), t.take(st * vt, vt), t.take(2 * st, 2), t.take(2 * st, 2)
  GF, EG, W, ENF, ONF, NK = t.take(1), t.take(1), t.take(st), t.take(st), t.take(st), t.take(st)
  OUT = t.outs()
  SS, K, VN, CC, EN, ON, CN = (t.take(kt * vt, vt), t.take(st * kt, kt), t.take(st * vt, vt), t.take(st), t.take(st),
                               t.take(st), t.take(8))
  E0, E0T, op = CN + 6, CN + 7, t.op  # ones in row 0, ones in column 0
  for j in range(st):  # end's row filled down, on's column across, and cum's, in parts for the FPU
    op(ENF + j, [(E0T, EN + j, False)]), op(ONF + j, [(ON + j, E0, False)])
    t.parts(CP, j, start=CC + j)
    for p in range(2): op(CF(j, p), [(CP(j, p), E0, False)])
  op(GF, [(ENF + j, CF(j, p), False) for j in range(st) for p in range(2)], exact=True)  # G in every place
  op(EG, start=GF, posts=[("fexp", ())], exact=True)
  for j in range(st):
    op(W + j, [(CP(j, p), E0, False) for p in range(2)], posts=[("rsub", (GF,)), ("min0", ()), ("fexp", ()), ("mul", (ONF + j,))],
       exact=True)
    t.norm(NK + j, K(j, 0), kt)
    for c in range(kt): op(KW(j, c), start=K(j, c), posts=[("mul", (NK + j,)), ("mul", (W + j,))])
    for v in range(vt): op(VR(j, v), start=VN(j, v))
  for c in range(kt):
    for j in range(st): op(KT(c, j), [(CN, KW(j, c), True)])
  for c in range(kt):
    for v in range(vt): op(OUT, [(KT(c, j), VR(j, v), False) for j in range(st)], posts=[("fma", (SS(c, v), EG))], home=(c, v))
  arg, get = {name: 6 + 4 * i for i, name in enumerate(["k", "vnew", "cum", "end", "on", "consts"])}, t.get
  fetch = (get(2, SS(0, 0), kt, vt) + get(arg["k"], K(0, 0), st, kt, f"e / {rep}") + get(arg["vnew"], VN(0, 0), st, vt) +
           get(arg["cum"], CC, st, 1) + get(arg["end"], EN, 1, st, f"e / {per}") + get(arg["on"], ON, st, 1, f"e / {per}") +
           f"for (unsigned t = 0; t < 8; t++) get(a[{arg['consts']}], {tile_of(arg['consts'], 't', 0, 0)}, {CN} + t);")
  return t.kernel("carry_kernel", fetch, kt * vt)

def carry(S, k, vnew, cum, end, on):
  """The state after a path through a piece that rule ran, as a prompt of the path would leave it, so nothing runs
  again (Steed's commit of a verified tree): S = e^G S + sum over the path of e^(G - cum_j) k_j v_new_j^T, where G is
  cum at the path's last token. k is as rule takes it, as the conv gives it, and v_new and cum as rule gives them; end is a row with 1 at the
  path's last token and on a column with 1 on its tokens, for each sequence of S's value heads. S is written over and
  returned. On the chip it's carry_kernel's one launch."""
  f, per = llama.fns(S), S.shape[0] // end.shape[0]
  if isinstance(S, np.ndarray):
    G, k = f.repeat(end, per) @ cum, unit(k)
    w = f.repeat(on, per) * np.exp(np.minimum(G - cum, 0))
    S[...] = S * np.exp(G) + f.t(f.repeat(k, S.shape[0] // k.shape[0]) * w) @ vnew
    return S
  key = ("carry", -(-vnew.rows // 32), -(-S.rows // 32), -(-S.cols // 32), S.batch // k.batch, per)
  return S.run(key, lambda: carry_kernel(*key[1:]), S, k, vnew, cum, end, on, constants(S.rt), units=S.batch, name="carry")

@functools.cache
def constants(rt):  # the identity, llama.pairs of 32 for spans 1 to 16, ones in row 0 and in column 0 (rule_kernel)
  return Tensor.of(rt, [np.eye(32).tolist()] + [llama.pairs(32, span).tolist() for span in (1, 2, 4, 8, 16)] +
                   [(e := np.pad(np.ones((1, 32)), ((0, 31), (0, 0)))).tolist(), e.T.tolist()], "fp32")

@functools.cache
def ones_row(rt): return Tensor.of(rt, [[1.0] * 32], "fp32")  # a row of ones, that a column of any length times fills

@functools.cache
def ones(rt): return Tensor.of(rt, np.ones((32, 32)).tolist(), "fp32")  # a tile of ones, that a column times fills

def moe(h, p, c, phases=lambda op: None, live=None):
  """A mixture of experts in place of the MLP, as in Qwen3 and Mixtral. The router picks the c.k best experts for each
  token (topk, on the chip). Their SwiGLUs are weighted by the softmax of their logits and summed, which is the same as
  the renormalized top-k probabilities of Qwen3. A shared expert, if the block has one, adds its SwiGLU, scaled by the
  sigmoid of shared_expert_gate.

  p has router (dim x experts), experts.gate_up and experts.down (the matrices of the experts as a batch), and the
  biases of gpt-oss: router.bias, experts.gate_up_bias and experts.down_bias, a row for each expert. phases(op) gives
  the fidelity phases of gate_up and down. c.limit is for the GLU of gpt-oss (glu). live is a column with 1 or 0 for
  each token. A token with 0 weights its experts by 0, so a decode step reads none of them for it (touched)."""
  f, norm = llama.fns(h), c.norm_topk
  logits = f.mm(h, p["router"], None)
  logits = logits + p["router.bias"] if "router.bias" in p else logits  # gpt-oss has a router bias
  values, ids = f.topk(logits if norm else f.softmax(logits), c.k)  # without norm_topk_prob, probabilities among all the experts
  w = f.softmax(values) if norm else values
  y = f.experts(h, p, ids, w if live is None else w * live, (phases("gate_up"), phases("down")), c.limit)
  if "shared_expert_gate" not in p: return y
  # Qwen2-MoE also has a shared expert, which all tokens go through
  return y + f.sigmoid(f.mm(h, p["shared_expert_gate"], None)) * llama.mlp(h, p, phases, "shared_expert.")

def numpy_experts(h, p, ids, w, phases, limit=None):  # the GLUs of the experts that each token chose, weighted by w and summed
  gate_up, down, m = p["experts.gate_up"], p["experts.down"], p["experts.down"].shape[-2]
  bias = lambda name, e: p[name][e] if name in p else 0
  def expert(x, e):
    g = x @ gate_up[e] + bias("experts.gate_up_bias", e)
    return llama.glu(g[:m], g[m:], limit) @ down[e] + bias("experts.down_bias", e)
  return np.stack([sum(w[t, j] * expert(h[t], e) for j, e in enumerate(ids[t])) for t in range(len(h))])

def chip_experts(h, p, ids, w, phases, limit=None):
  """Run the selected experts on the chip. Indexed matmuls read expert weights where they are stored.

  For up to 128 tokens, touched creates a batch entry for each selected expert. Tokens become rows of each
  entry, with zero routing weight for tokens that didn't select that expert. Multiply the SwiGLU outputs by
  these weights and sum across entries. Workers read the token rows once and share each expert's weight
  reads.

  For longer prompts, group sorts tokens by expert and pads each group to 32 rows. Each tile becomes a batch
  entry, so weights are read once per 32 tokens. Gather the k outputs per token, then weight and sum them.
  Above 128 tokens, choose the layout with fewer tile rows. Both layouts support replay; unused entries
  produce zeros. Gather expert biases using the same indices."""
  gate_up, down = p["experts.gate_up"], p["experts.down"]
  (n, k), m, e = ids.shape[1:], down.rows, gate_up.batch
  dense = n <= 128 or -(-n // 32) * min(n * k, e) <= -(-n * k // 32) + min(n * k, e)  # tile rows per matrix, each way
  if dense: (picks, weights), x = touched(ids, e, w), h
  else:
    order, picks, places = group(ids, e)
    x = h.gather(order).split(picks.rows, 1)
  entries = picks.rows
  row = picks.reshape(1, entries).minimum(e - 1) if {"experts.gate_up_bias", "experts.down_bias"} & set(p) else None  # the picks as a row
  # the bias of the expert of each entry. An entry with no expert gets the bias of the last expert, weighted 0
  bias = lambda name: p[name].gather(row).reshape(entries, 1, p[name].cols)
  plus = lambda y, name, by=1: y + bias(name) * by if name in p else y
  x = plus(x.matmul(gate_up, fidelity=phases[0], index=picks), "experts.gate_up_bias")
  g = llama.glu(x[:, :, :m], x[:, :, m:], limit)
  if dense: return plus((g * weights).matmul(down, "fp32", phases[1], picks), "experts.down_bias", weights).sum(0)
  y = plus(g.matmul(down, "fp32", phases[1], picks), "experts.down_bias").reshape(order.cols, down.cols)
  run = places.cols // k  # the length of the run of places for each choice
  terms = [y.gather(places[:, :, j * run:j * run + n]) * w[:, :, j:j + 1] for j in range(k)]
  return sum(terms[1:], terms[0])

# The tiles of entries that group needs for any choices: a tile for each 32 choices, plus one for each value that can
# come, since its run may end partway through a tile. Also the length of a run of places, the rows rounded up to a tile.
def group_sizes(rows, k, n): return -(-rows * k // 32) + min(n, rows * k), -(-rows // 32) * 32

def group_kernel(rows, k, n):
  """Sort token choices by expert on one BRISC worker. Arguments are [first unit, units, order address, tiles
  address, places address, choices address, choices row tile stride, choices tile offset].

  Read rows x k choices with values below n. Count each expert, assign consecutive entries, and pad each
  expert's group to 32. Build order (token per entry), tiles (expert per tile) and places (entry per choice,
  grouped by choice index) as flat arrays in L1. Send them as tiles, repeating rows or columns to match
  tensor storage."""
  t, rp = group_sizes(rows, k, n)
  fills = DATA + -(-rows // 32) * 4096  # after the choices: the count of each value, then its next entry
  out = (fills + 4 * (n + 33 * t + k * rp) + 63) // 64 * 64
  if out + 4096 > OUTS: raise ValueError("group_kernel: the choices and their arrays don't fit in L1")
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define FILL {fills:#x}
#define ORDER (FILL + {4 * n})
#define TILES (ORDER + {4 * 32 * t})
#define PLACES (TILES + {4 * t})
#define OUT {out:#x}
{ROWS}
static unsigned choice(unsigned r, unsigned j) {{ return ((volatile unsigned *)IDS)[r / 32 * 1024 + at(r % 32, j)]; }}
static void send(unsigned base, unsigned i) {{ noc_write(OUT, BANK(i), AT(base, i, 4096), 4096); noc_wait(); }}
void kmain(volatile unsigned *a) {{
  volatile unsigned *fill = (volatile unsigned *)FILL, *order = (volatile unsigned *)ORDER;
  volatile unsigned *tiles = (volatile unsigned *)TILES, *places = (volatile unsigned *)PLACES, *out = (volatile unsigned *)OUT;
  if (!a[1]) return;
  for (unsigned i = 0; i < {-(-rows // 32)}; i++) {{ unsigned u = a[7] + i * a[6]; noc_get(BANK(u), AT(a[5], u, 4096), IDS + i * 4096, 4096); }}
  for (unsigned e = 0; e < {n}; e++) fill[e] = 0;
  for (unsigned s = 0; s < {32 * t}; s++) order[s] = 0;
  for (unsigned i = 0; i < {t}; i++) tiles[i] = {n};  // n marks unused entries, which indexed matmul skips
  for (unsigned s = 0; s < {k * rp}; s++) places[s] = 0;
  noc_wait();
  for (unsigned r = 0; r < {rows}; r++) for (unsigned j = 0; j < {k}; j++) fill[choice(r, j)]++;
  for (unsigned e = 0, s = 0; e < {n}; e++) {{  // the run of each value: its count rounded up to a multiple of 32, after the runs before it
    unsigned count = fill[e];
    fill[e] = s;
    for (unsigned end = s + (count + 31) / 32 * 32; s < end; s += 32) tiles[s / 32] = e;
  }}
  for (unsigned r = 0; r < {rows}; r++)
    for (unsigned j = 0; j < {k}; j++) {{ unsigned s = fill[choice(r, j)]++; order[s] = r; places[j * {rp} + r] = s; }}
  for (unsigned i = 0; i < {t}; i++) {{  // order and places are rows: fill row 0 of each tile and copy it down. tiles is a column
    for (unsigned c = 0; c < 32; c++) out[at(0, c)] = order[i * 32 + c];
    row_down(OUT, 4), send(a[2], i);
  }}
  for (unsigned i = 0; i < {-(-t // 32)}; i++) {{
    for (unsigned r = 0; r < 32; r++) for (unsigned c = 0; c < 32; c++) out[at(r, c)] = i * 32 + r < {t} ? tiles[i * 32 + r] : {n};
    send(a[3], i);
  }}
  for (unsigned i = 0; i < {k * rp // 32}; i++) {{
    for (unsigned c = 0; c < 32; c++) out[at(0, c)] = places[i * 32 + c];
    row_down(OUT, 4), send(a[4], i);
  }}
}}""")

def group(ids, n):
  """Group rows x k int32 choices by expert, with IDs below n. Return (order, tiles, places), all int32.
  group_kernel runs this on BRISC.

  order is a row of 32*t entries, with t from group_sizes. Each entry holds the source token row, or zero for
  padding. Expert groups start on tile boundaries. tiles is a column of t expert IDs, with n for unused
  tiles. places has k runs, one entry index per token per choice, with each run padded to a tile. Indexed
  matmul uses tiles to select the expert weights."""
  if not (ids.dtype == "int32" and ids.batch == 1 and ids.cols <= 32):
    raise ValueError("group needs ids as a matrix of int32 choices, at most 32 to a row")
  x, rows, k = ids.realize(), ids.rows, ids.cols  # a matrix, so there's no repeat to copy
  t, rp = group_sizes(rows, k, n)
  order, tiles, places = Tensor(ids.rt, 1, 32 * t, dtype="int32"), Tensor(ids.rt, t, 1, dtype="int32"), Tensor(ids.rt, 1, k * rp, dtype="int32")
  order.run(("group", rows, k, n), lambda: group_kernel(rows, k, n), order.addr, tiles.addr, places.addr, x.addr, x.stride, x.offset,
            units=1, name="group")
  return order, tiles, places

def touched_l1(rows, k, n):
  """The L1 address of touched's output tile, after its ids, weights and expert list. Reject inputs that would put
  the end of the tile past OUTS."""
  if type(n) is not int or n < 1: raise ValueError("touched needs a positive integer expert count")
  out = (DATA + -(-rows // 32) * 6144 + 4 * -(-n // 32) + 4 * min(rows * k, n) + 63) // 64 * 64
  if out + 4096 > OUTS: raise ValueError(f"touched inputs and expert list do not fit L1: {rows} rows, {k} choices, {n} experts")
  return out

def touched_kernel(rows, k, n):
  """The kernel for touched, which lists the experts that some row picks. It runs on BRISC alone, with a unit for each
  entry.

  Arguments: first unit, units, the addresses of experts and weights, then the address, tile stride for rows and tile
  offset of the choices, then the same for the bf16 weights of the choices.
  Workers read the choices (rows x k, values below n) and their weights once. They mark the experts picked with a
  nonzero weight in a bitmask, and list them in increasing order, then n (none) after them. Entry s writes its tile
  of each tile row of its weights: row r holds the weight that row r gave expert s of the list (0 if it didn't pick
  it), across the row. The first entry of each 32 also writes its tile of the experts column, each expert across its
  row, as the tile of a column holds it."""
  out = touched_l1(rows, k, n)
  tr, entries = -(-rows // 32), min(rows * k, n)
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define WEIGHTS {DATA + tr * 4096:#x}
#define LIST {DATA + tr * 6144:#x}
#define OUT {out:#x}
{ROWS}
// non-volatile reads of L1 that the NoC wrote. They're safe after the fence in noc_wait
static unsigned choice(unsigned r, unsigned j) {{ return ((const unsigned *)IDS)[r / 32 * 1024 + at(r % 32, j)]; }}
static unsigned weight(unsigned r, unsigned j) {{ return ((const unsigned short *)WEIGHTS)[r / 32 * 1024 + at(r % 32, j)]; }}
// the position of the lowest set bit of m. rv32im has no instruction to count zeros, so this uses a de Bruijn sequence
static unsigned lowest(unsigned m) {{
  static const unsigned char place[32] = {{0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26,
                                           12, 18, 6, 11, 5, 10, 9}};
  return place[(m & -m) * 0x077CB531u >> 27];
}}
void kmain(volatile unsigned *a) {{
  unsigned *list = (unsigned *)LIST, *mark = list + {entries}, *out = (unsigned *)OUT;
  if (!a[1]) return;
  for (unsigned i = 0; i < {tr}; i++) {{
    unsigned u = a[6] + i * a[5], v = a[9] + i * a[8];
    noc_get(BANK(u), AT(a[4], u, 4096), IDS + i * 4096, 4096), noc_get(BANK(v), AT(a[7], v, 2048), WEIGHTS + i * 2048, 2048);
  }}
  copy({ZEROS:#x}, OUT, 4096);  // rows of a weights tile past the choices stay 0
  for (unsigned w = 0; w < {-(-n // 32)}; w++) mark[w] = 0;
  noc_wait();
  for (unsigned r = 0; r < {rows}; r++)
    for (unsigned j = 0; j < {k}; j++) if (weight(r, j)) {{ unsigned e = choice(r, j); mark[e / 32] |= 1u << e % 32; }}
  unsigned count = 0;
  for (unsigned w = 0; w < {-(-n // 32)}; w++) for (unsigned m = mark[w]; m; m &= m - 1) list[count++] = w * 32 + lowest(m);
  for (; count < {entries}; count++) list[count] = {n};
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned s = a[0] + u, e = list[s];
    for (unsigned i = 0; i < {tr}; i++) {{  // tile row i of its weights: the weight of each row for e, across the row
      for (unsigned r = 0; r < 32 && i * 32 + r < {rows}; r++) {{
        unsigned w = 0;
        for (unsigned j = 0; j < {k}; j++) if (choice(i * 32 + r, j) == e) w = weight(i * 32 + r, j);
        for (unsigned *row = out + at(r, 0) / 2, c = 0; c < 8; c++) row[c] = row[128 + c] = w | w << 16;  // its runs in both faces
      }}
      unsigned t = s * {tr} + i;
      l1_settle(OUT);
      noc_write(OUT, BANK(t), AT(a[3], t, 2048), 2048), noc_wait();
    }}
    if (s % 32) continue;
    // the tile of the experts column for entries s to s + 31, each expert across its row
    for (unsigned r = 0; r < 32; r++) for (unsigned *row = out + at(r, 0), e = s + r < {entries} ? list[s + r] : {n}, c = 0; c < 16; c++)
      row[c] = row[256 + c] = e;
    l1_settle(OUT);
    noc_write(OUT, BANK(s / 32), AT(a[2], s / 32, 4096), 4096), noc_wait();
    copy({ZEROS:#x}, OUT, 4096), noc_wait();
  }}
}}""")

def touched(ids, n, weights):
  """List distinct experts selected with nonzero weight. ids is an int32 (rows, k) matrix with values
  below n; weights holds the corresponding bf16 routing weights. touched_kernel runs this on BRISC.
  Return:
  - experts: a column of min(rows * k, n) entries, sorted by expert ID and padded with n.
  - weights: bf16 (entries, rows, 1), with zero where a token didn't select that expert.

  Indexed matmul then runs each selected expert over the token rows. Padding entries select no
  matrix and return zeros. Weight the results and sum across experts."""
  if not (ids.dtype == "int32" and weights.dtype == "bf16" and ids.shape == weights.shape and ids.batch == 1 and ids.cols <= 32):
    raise ValueError(f"touched needs int32 ids and bf16 weights of one shape, at most 32 choices to a row: {ids.shape} {weights.shape}")
  touched_l1(ids.rows, ids.cols, n)
  if ids.rt is not weights.rt: raise ValueError("touched needs ids and weights on the same runtime")
  x, w, rows, k = ids.realize(), weights.realize(), ids.rows, ids.cols  # matrices, so there are no repeats to copy
  entries = min(rows * k, n)
  experts, out = Tensor(ids.rt, entries, 1, dtype="int32"), Tensor(ids.rt, rows, 1, batch=entries)
  experts.run(("touched", rows, k, n), lambda: touched_kernel(rows, k, n), experts.addr, out.addr, x.addr, x.stride, x.offset,
              w.addr, w.stride, w.offset, units=entries, name="touched")
  return experts, out

def mxfp4(blocks, scales):
  """Convert gpt-oss MXFP4 expert weights to bf16 bits exactly; each value has at most two significant bits.

  The values are 4 bits (sign, 2 bits of exponent, 1 of mantissa), two to a byte with the low nibble first. Blocks of 32
  share a power of 2, stored as a byte with 127 for 1. They come as a batch of (out x in) matrices, which are
  transposed for x @ w."""
  values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
  v = np.stack([values[blocks & 15], values[blocks >> 4]], -1).reshape(*scales.shape, 32)
  v = np.ldexp(v, scales.astype(np.int32)[..., None] - 127).reshape(*scales.shape[:-1], -1)
  return gpt2.bf16_bits(np.ascontiguousarray(np.swapaxes(v, -1, -2)))

def draft(ids, places, h, w, c, caches, slots=None, length=None, every=True):
  """Predict one token beyond ids using the multi-token prediction block at c.layers. h is the hidden state at
  places; ids contains the following token. Return logits for the token after ids.

  RMS-normalize the embedding and h, concatenate them for fc, then run the block, norm and output layer.
  Update caches[c.layers] at places. slots and length follow llama.forward. With every=False, return only
  the last position, as needed after a prompt chunk."""
  f = llama.fns(h)
  rows, rotary, mask, _, live = llama.around(w, c, places, caches[c.layers].shape[-2], slots, length)
  g = predict(ids, h, w, c, caches[c.layers], rows, rotary, mask, slots is None and h.shape[-2] > 1, live, slots)
  own = "embed.T draft" in w  # a copy of the output layer for the draft, when the precision has a "draft head"
  last = g if every else f.rows(g, [g.shape[-2] - 1])
  return f.mm(last, w["embed.T draft" if own else "embed.T"], llama.precision_of(c, "draft head" if own else "head")[1], "fp32")

def predict(ids, h, w, c, cache, rows, rotary, mask, prefill, live=None, slots=None):
  """The multi-token prediction block's normalized output for tokens ids after hidden states h (draft's pass): fc
  reads the normalized embedding and h side by side, and the block writes its keys and values into cache at rows."""
  f, l = llama.fns(h), c.layers
  e, h = llama.rms(f.rows(w["embed"], ids), w["mtp.pre_fc_norm_embedding"], c), llama.rms(h, w["mtp.pre_fc_norm_hidden"], c)
  eh = np.concatenate([e, h], -1) if isinstance(h, np.ndarray) else e.cat(h, axis=-1)  # side by side, one matmul's A
  x = f.mm(eh, w["mtp.fc"], llama.precision_of(c, "fc")[1], "fp32")
  x = llama.layer(x, gpt2.block(w, f"layers.{l}."), c, rotary, mask, cache, rows, l, prefill, live, slots=slots)
  return llama.rms(x, w["mtp.norm"], c)
