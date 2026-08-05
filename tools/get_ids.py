from tokenizers import Tokenizer
import sys
t = Tokenizer.from_file("D:/models/LFM2.5-2.6B/tokenizer.json")
prompt = sys.argv[1] if len(sys.argv) > 1 else "The future of artificial intelligence is"
ids = t.encode(prompt).ids
print("IDS", ids)
# also emit as space-separated for direct C feeding
print("SPACESEP", " ".join(str(i) for i in ids))
