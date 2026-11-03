# The first workload's tokenizer gate (docs/workload.md). For each model folder, baretile's tokenizer must give the same
# token IDs as Hugging Face's tokenizer from the same folder, with add_special_tokens=False, and decode them to the same
# text, for the workload's three prompts and a text of accents, digits and whitespace. It prints each difference, and
# exits with an error if there is one.
#   python3 test/tokenizer.py path/to/SmolLM-135M [path/to/Llama-3.1-8B-Instruct ...]
# It needs transformers.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile.models import llama
from generation import PROMPTS

TEXTS = PROMPTS + ("  café İstanbul 12345\n\tHello!",)

def check(folder, texts=TEXTS):  # the differences between the two tokenizers on texts, as lines to print
  import transformers
  ours, theirs, out = llama.Tokenizer(folder), transformers.AutoTokenizer.from_pretrained(folder), []
  for text in texts:
    a, b = ours.encode(text), theirs.encode(text, add_special_tokens=False)
    if a != b: out.append(f"{text!r}: baretile's IDs {a}, Hugging Face's {b}")
    elif (x := ours.decode(a)) != (y := theirs.decode(b)): out.append(f"{text!r}: baretile decodes {x!r}, Hugging Face {y!r}")
  return out

if __name__ == "__main__":
  if len(sys.argv) < 2: sys.exit("usage: python3 test/tokenizer.py model-folder [model-folder ...]")
  import transformers
  print(f"transformers {transformers.__version__}")
  failed = False
  for folder in sys.argv[1:]:
    differences = check(folder)
    print(f"{folder}: " + (f"{len(differences)} of {len(TEXTS)} texts differ" if differences else f"all {len(TEXTS)} texts match"))
    for line in differences: print("  " + line)
    failed |= bool(differences)
  sys.exit(1 if failed else 0)
