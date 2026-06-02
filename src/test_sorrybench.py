from datasets import load_dataset

# load SorryBench
ds = load_dataset("sorry-bench/sorry-bench-202503")

# inspect dataset
print(ds)

print("\nFirst sample:")
print(ds["train"][0])

