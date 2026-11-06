# The first workload's timing protocol and its single-slot HTTP gate (docs/workload.md), each as one command.
#   python3 test/workload.py time path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [report.json] [runs] [steps]
#   python3 test/workload.py http path/to/Llama-3.1-8B-Instruct [report.json] [tokens]
# time runs the protocol in fresh processes, 3 unless runs says otherwise. Each loads the model at tt precision, with
# its bfp8 KV cache, uploads it, prefills the reference's first 512 tokens and picks the first token, then takes 200
# greedy steps, feeding each pick back and ignoring EOS. Steps 0 and 1 run and record the jit, so the rate counts the
# other 198, each from the token on the host to the next one picked there from the logits read back. Loading,
# uploading and the prompt's first token are timed on their own. The report keeps each process's results, whether its
# kernels were compiled already, and their median, with what workload.md asks to record: the device, its PCIe link,
# firmware, tt-kmd and workers, the host, the package versions, the model files' hashes, baretile's commit, the
# precision and the command. On ttsim the times are ttsim's clock, so they only check the script.
# http starts serve.py with one slot of 1,024 positions at tt with a bfp8 KV cache, and sends each of the three prompts
# at temperature 0, without and then with streaming, to up to 32 tokens, stopping at EOS. Each answer needs HTTP 200,
# some text, a stop or length finish, usage that counts the prompt's tokens and no more tokens than asked, and the same
# text and counts in both forms. Then a stream that's cancelled after its first text must give the slot back, so that
# the next request finishes. Both exit with an error on any miss, and keep it in the report, http with the server's
# last lines of output.
import hashlib, http.client, json, os, pathlib, re, statistics, subprocess, sys, tempfile, threading, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import benchmark, generation
from baretile import bh, pci
from baretile.models import llama

PROMPT, STEPS, RUNS, TIMED_FROM = 512, 200, 3, 2  # the reference tokens prefilled, the decode steps, the processes, the first timed step
SAMPLING = dict(temperature=0, top_k=0, top_p=1, min_p=0, repetition_penalty=1, presence_penalty=0, frequency_penalty=0)
DEADLINE, RETRIES = 3600, 30  # seconds for a process or a request, and seconds to wait for a cancelled stream's slot

def one(folder, reference, steps):
  """One process's run of the timing protocol, as a dict of its seconds, its rate and its picks, and the device it ran
  on. The prompt's first token is timed from its ids on the host, as each step is."""
  from baretile.chip import Chip
  from baretile.runtime import CACHE, Runtime
  from baretile.tensor import Tensor, jit
  images, start = set(CACHE.glob("*")) if CACHE.exists() else set(), time.time()
  prompt = llama.reference(reference)[0][:PROMPT]
  size = -(-(len(prompt) + steps + 1) // 32) * 32  # 736 rows for 200 steps, as B2's decode allocates
  c, weights = llama.load(folder, size)
  c.precision = llama.precision(c, "tt")
  loaded, chip = time.time(), Chip(pci.open_device())
  rt, clock, sim = Runtime(chip), chip.pci.now, isinstance(chip.pci, pci.Sim)
  began = clock()
  t = llama.upload(rt, weights, c.precision)
  rt.wait()
  uploaded, on = clock(), lambda row: Tensor.of(rt, [row], "int32")
  kv = llama.kv_cache(c, size, llama.zeros(rt, llama.cache_format(c)))
  picks = [int(llama.logits(llama.forward(on(prompt), on(list(range(len(prompt)))), t, c, kv)).argmax())]
  first, step, times = clock(), jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), []
  for i in range(steps):
    at = clock()
    picks.append(int(llama.logits(step(on(picks[-1:]), on([len(prompt) + i]))).argmax()))
    times.append(clock() - at)
  return dict(load_s=loaded - start, upload_s=uploaded - began, first_token_s=first - uploaded, step_s=times,
              tokens_per_s=(steps - TIMED_FROM) / sum(times[TIMED_FROM:]), picks=picks, kv=llama.cache_format(c),
              compiled=len((set(CACHE.glob("*")) if CACHE.exists() else set()) - images), workers=len(rt.cores),
              card=benchmark.card(sim), clock="ttsim's" if sim else "the host's", aiclk_mhz=chip.aiclk,
              firmware=".".join(str(chip.telemetry().get(bh.TAG_FIRMWARE, 0) >> b & 255) for b in (24, 16, 8)))

def hashes(folder):  # sha256 of each file in the model's folder, as docs/workload.sha256 lists them
  out = {}
  for p in sorted(pathlib.Path(folder).iterdir()):
    if p.is_file():
      h = hashlib.sha256()
      with open(p, "rb") as f:
        while chunk := f.read(1 << 24): h.update(chunk)
      out[p.name] = h.hexdigest()
  return out

def versions():  # the packages a run took part with, and tt-kmd's version where it drives the card
  out = {"baretile": generation.commit(), **benchmark.host()}
  for name in ("torch", "transformers"):
    try: out[name] = __import__(name).__version__
    except ImportError: out[name] = "not installed"
  kmd = pathlib.Path(pci.MODULE, "version")
  out["tt-kmd"] = kmd.read_text().strip() if kmd.exists() else "not loaded"
  return out

def timing(folder, reference, runs=RUNS, steps=STEPS):
  """Run the protocol in runs fresh processes, and return the report with each run, their medians and whether every
  process ran."""
  record, failures, folder, reference = [], [], os.path.abspath(folder), os.path.abspath(reference)  # the processes run in ROOT
  for i in range(runs):
    with tempfile.NamedTemporaryFile(suffix=".json") as out:
      done = subprocess.run([sys.executable, os.path.abspath(__file__), "one", folder, reference, str(steps), out.name], cwd=ROOT,
                            capture_output=True, text=True, timeout=DEADLINE)
      if done.returncode: failures.append(dict(run=i, exit=done.returncode, output=(done.stdout + done.stderr)[-4000:]))
      else: record.append(json.load(open(out.name)))
  median = {k: statistics.median(r[k] for r in record) for k in ("load_s", "upload_s", "first_token_s", "tokens_per_s")} if record else {}
  same = len({tuple(r["picks"]) for r in record}) <= 1  # greedy picks don't change between processes
  if not same: failures.append(dict(problem="the runs picked different tokens"))
  return dict(passed=not failures and len(record) == runs, runs=record, median=median, failures=failures, steps=steps, timed=steps - TIMED_FROM,
              prompt_tokens=PROMPT, precision="tt", versions=versions(), model=hashes(folder), command=sys.argv)

def post(port, body, cancel=False):
  """POST body to /v1/completions. Return (status, the answer) for a complete response, or the text, finish reason and
  usage gathered from a stream's chunks, with the seconds to its first text. With cancel, hang up after the first text."""
  conn, start = http.client.HTTPConnection("127.0.0.1", port, timeout=DEADLINE), time.time()
  conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
  reply = conn.getresponse()
  if not body.get("stream"):
    data = reply.read()
    conn.close()
    return reply.status, json.loads(data) if data else {}
  text, finish, usage, first = "", None, None, None
  for line in reply:
    line = line.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]": continue
    chunk = json.loads(line.removeprefix("data: "))
    for choice in chunk.get("choices", []):
      if choice.get("text") and first is None: first = time.time() - start
      text, finish = text + choice.get("text", ""), choice.get("finish_reason") or finish
    usage = chunk.get("usage") or usage
    if cancel and first is not None: break
  conn.close()
  return reply.status, dict(text=text, finish_reason=finish, usage=usage, first_text_s=first)

def gate(folder, tokens=32):
  """Start serve.py with one slot of 1,024 positions at tt with a bfp8 cache, run the single-slot HTTP gate against it,
  and return the report."""
  folder = os.path.abspath(folder)  # the server runs in ROOT
  server = subprocess.Popen([sys.executable, "-m", "baretile.models.serve", folder, "0", "1", "1024", "tt", "bfp8"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ | {"PYTHONPATH": ROOT})
  slow, lines = threading.Timer(DEADLINE, server.kill), []  # a server that hangs before it's ready ends its output, and so the loop
  slow.start()
  try:
    for line in server.stdout:
      lines.append(line)
      if m := re.search(r"on port (\d+)", line): break
    else: return dict(passed=False, failures=[dict(problem="the server stopped before it was ready", output="".join(lines)[-4000:])])
    slow.cancel()
    threading.Thread(target=lambda: [lines.append(l) for l in server.stdout], daemon=True).start()  # keep reading, so it never blocks
    port, tok, prompts, failures = int(m[1]), llama.Tokenizer(folder), [], []
    for prompt in generation.PROMPTS:
      body = dict(prompt=prompt, max_tokens=tokens, **SAMPLING)
      (plain_status, plain), (stream_status, streamed) = post(port, body | dict(stream=False)), \
        post(port, body | dict(stream=True, stream_options=dict(include_usage=True)))
      answer = (plain.get("choices") or [{}])[0]
      usage, want, made = plain.get("usage") or {}, len(tok.encode(prompt)), (plain.get("usage") or {}).get("completion_tokens", 0)
      # a reused prefix changes only cached_tokens, so the forms compare their counts
      counts = lambda u: {k: (u or {}).get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
      checks = {"HTTP 200": plain_status == stream_status == 200, "text": bool(answer.get("text")),
                "finish": answer.get("finish_reason") in ("stop", "length") and streamed["finish_reason"] == answer.get("finish_reason"),
                "usage": usage.get("prompt_tokens") == want and (made == tokens if answer.get("finish_reason") == "length" else 0 < made <= tokens),
                "same in both forms": streamed["text"] == answer.get("text") and counts(streamed["usage"]) == counts(usage)}
      prompts.append(dict(prompt=prompt, text=answer.get("text"), finish_reason=answer.get("finish_reason"), usage=usage,
                          first_text_s=streamed["first_text_s"], checks=checks))
      failures += [dict(prompt=prompt, check=k) for k, ok in checks.items() if not ok]
    cancelled, _ = post(port, dict(prompt=generation.PROMPTS[0], max_tokens=tokens, stream=True, **SAMPLING), cancel=True)
    for _ in range(RETRIES):  # the server frees a slot once it notices the hang-up, within a second (test_serve_hangup)
      status, after = post(port, dict(prompt=generation.PROMPTS[1], max_tokens=tokens, stream=False, **SAMPLING))
      if status != 429: break
      time.sleep(1)
    recovered = cancelled == status == 200 and bool((after.get("choices") or [{}])[0].get("text"))
    if not recovered: failures.append(dict(check="a cancelled stream gives its slot back", status=status, answer=after))
    return dict(passed=not failures, prompts=prompts, recovered=recovered, failures=failures, max_tokens=tokens, sampling=SAMPLING,
                server="".join(lines[-40:]), versions=versions(), model=hashes(folder), command=sys.argv)
  finally:
    slow.cancel()
    server.kill()
    server.wait()

def main():
  usage = ("usage: python3 test/workload.py time model reference.refpt [report.json] [runs] [steps]\n"
           "       python3 test/workload.py http model [report.json] [tokens]")
  if len(sys.argv) < 3 or sys.argv[1] not in ("time", "http", "one"): sys.exit(usage)
  if sys.argv[1] == "one":  # a fresh process of time's: one run, written to the file named last
    return json.dump(one(sys.argv[2], sys.argv[3], int(sys.argv[4])), open(sys.argv[5], "w"))
  if sys.argv[1] == "time":
    if len(sys.argv) < 4: sys.exit(usage)
    rest = sys.argv[4:]
    report = timing(sys.argv[2], sys.argv[3], int(rest[1]) if len(rest) > 1 else RUNS, int(rest[2]) if len(rest) > 2 else STEPS)
    summary = (f"{len(report['runs'])} runs, median {report['median'].get('tokens_per_s', 0):.2f} tokens/s over {report['timed']} steps, "
               f"first token {report['median'].get('first_token_s', 0):.2f}s, upload {report['median'].get('upload_s', 0):.1f}s")
  else:
    rest = sys.argv[3:]
    report = gate(sys.argv[2], int(rest[1]) if len(rest) > 1 else 32)
    passed = sum(all(p["checks"].values()) for p in report.get("prompts", []))
    kept = "gave back" if report.get("recovered") else "kept"
    summary = f"{passed} of {len(generation.PROMPTS)} prompts passed, and a cancelled stream {kept} its slot"
  path = rest[0] if rest else f"workload-{sys.argv[1]}.json"
  json.dump(report, open(path, "w"), indent=1)
  print(f"{'passed' if report['passed'] else 'FAILED'}: {summary}; the report is in {path}")
  for f in report["failures"]: print(f"  {json.dumps(f)[:400]}")
  sys.exit(0 if report["passed"] else 1)

if __name__ == "__main__": main()
