from collections import Counter
import json

with open("data/raw/sorrybench_base.json", "r") as f:
    data = json.load(f)

categories = Counter()

for item in data:
    categories[item["category"]] += 1

print(categories)