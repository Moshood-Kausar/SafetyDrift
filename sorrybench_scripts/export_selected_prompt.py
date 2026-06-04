from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "mapping_methods" / "M5_specmap_top3.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "selected_sorrybench_prompt_manifest.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep only prompt-level coverage mappings.
    df = df[df["coverage_label"].isin([
        "full_coverage",
        "partial_coverage_same_risk_domain",
    ])].copy()

    manifest = df[
        [
            "prompt_id",
            "question_id",
            "sorrybench_category",
            "claim_category",
            "coverage_label",
        ]
    ].drop_duplicates(subset=["prompt_id"]).copy()

    manifest = manifest.sort_values(["claim_category", "question_id"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote manifest: {OUTPUT_PATH}")
    print(f"Unique prompts: {len(manifest)}")
    print()
    print(manifest["claim_category"].value_counts())


if __name__ == "__main__":
    main()