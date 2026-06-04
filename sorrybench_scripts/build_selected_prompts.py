from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "selected_sorrybench_prompt_manifest.csv"
SORRYBENCH_PATH = PROJECT_ROOT / "data" / "sorrybench.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "selected_prompts_for_mutation.csv"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    if not SORRYBENCH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SORRYBENCH_PATH}. "
            "Run your SORRY-Bench loading script first to create data/sorrybench.csv."
        )

    sorry = pd.read_csv(SORRYBENCH_PATH)
    sorry = sorry[sorry["prompt_style"] == "base"].copy()

    selected = manifest.merge(
        sorry[["prompt_id", "question_id", "category", "prompt_style", "prompt"]],
        on=["prompt_id", "question_id"],
        how="left",
    )

    missing = selected[selected["prompt"].isna()]
    if len(missing) > 0:
        print("ERROR: Some selected prompts were not found in data/sorrybench.csv")
        print(missing[["prompt_id", "question_id"]])
        raise SystemExit(1)

    selected = selected[
        [
            "prompt_id",
            "question_id",
            "claim_category",
            "sorrybench_category",
            "coverage_label",
            "category",
            "prompt_style",
            "prompt",
        ]
    ].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote selected prompts: {OUTPUT_PATH}")
    print(f"Rows: {len(selected)}")
    print()
    print("Prompt counts by claim category:")
    print(selected["claim_category"].value_counts())


if __name__ == "__main__":
    main()