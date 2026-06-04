from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATASET_NAME = "sorry-bench/sorry-bench-202406"
OUTPUT_PATH = "data/sorrybench.csv"


def main():
    ds = load_dataset(DATASET_NAME)
    df = ds["train"].to_pandas()

    rows = []

    for _, row in df.iterrows():
        turns = row["turns"]

        # turns is a list with one prompt
        if isinstance(turns, list):
            prompt = turns[0]
        else:
            prompt = str(turns)

        rows.append(
            {
                "prompt_id": f"sb_{int(row['question_id']):04d}_{row['prompt_style']}",
                "question_id": int(row["question_id"]),
                "category": str(row["category"]),
                "prompt_style": str(row["prompt_style"]),
                "prompt": prompt,
            }
        )

    out = pd.DataFrame(rows)

    Path("data").mkdir(exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")

    print("\nPrompt styles:")
    print(out["prompt_style"].value_counts())

    print("\nBase prompt category counts:")
    base = out[out["prompt_style"] == "base"]
    print(base["category"].value_counts().sort_index())


if __name__ == "__main__":
    main()