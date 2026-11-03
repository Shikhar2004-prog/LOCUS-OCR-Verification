import sys
from datasets import load_dataset
name, out = sys.argv[1], sys.argv[2]
ds = load_dataset(name)
print(name, ds)
ds.save_to_disk(out)
