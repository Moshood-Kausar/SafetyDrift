from datasets import load_dataset

dataset_name = "sorry-bench/sorry-bench-202406"

ds = load_dataset(dataset_name)

print(ds)
print("\nSplits:", ds.keys())

for split_name in ds.keys():
    print(f"\n--- Split: {split_name} ---")
    print(ds[split_name])
    print("Columns:", ds[split_name].column_names)
    print("First row keys:", ds[split_name][0].keys())
    print("First row without prompt text:")
    row = dict(ds[split_name][0])
    for key in row:
        if "prompt" in key.lower() or "instruction" in key.lower() or "question" in key.lower():
            row[key] = "[PROMPT HIDDEN]"
    print(row)