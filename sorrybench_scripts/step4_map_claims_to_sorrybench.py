import pandas as pd
from pathlib import Path


CATEGORY_MAP = {
    "fairness_bias": "2",
    "mental_health_self_harm": "6",
    "harmful_violent_content": "7",
}


def main():
    claims_path = "../outputs/annotated_claims.csv"
    sorrybench_path = "../data/sorrybench.csv"
    output_path = "../outputs/mapped_claims_to_prompts.csv"

    claims = pd.read_csv(claims_path)
    sorry = pd.read_csv(sorrybench_path)

    # Only use original SorryBench prompts.
    # Do not use SorryBench's built-in mutations, because we generate our own later.
    sorry = sorry[sorry["prompt_style"] == "base"].copy()

    testable_claims = claims[
        (claims["label_claim"] == "Claim")
        & (claims["label_testable"] == "Testable")
        & (claims["category"].notna())
    ].copy()

    print(f"Total annotated rows: {len(claims)}")
    print(f"Testable claims kept: {len(testable_claims)}")
    print(f"SorryBench base prompts loaded: {len(sorry)}")

    rows = []

    for _, claim_row in testable_claims.iterrows():
        claim_category_name = str(claim_row["category"]).strip()
        sorrybench_category = CATEGORY_MAP.get(claim_category_name)

        if sorrybench_category is None:
            print(f"WARNING: no category mapping for {claim_category_name}")
            continue

        matching_prompts = sorry[
            sorry["category"].astype(str).str.strip() == sorrybench_category
        ]

        if matching_prompts.empty:
            print(
                f"WARNING: no SorryBench prompts found for "
                f"{claim_category_name} -> {sorrybench_category}"
            )
            continue

        for _, prompt_row in matching_prompts.iterrows():
            rows.append(
                {
                    "claim_id": claim_row["id"],
                    "source_doc": claim_row["source_doc"],
                    "claim": claim_row["claim_text"],
                    "label_claim": claim_row["label_claim"],
                    "label_testable": claim_row["label_testable"],
                    "claim_category": claim_category_name,
                    "sorrybench_category": sorrybench_category,
                    "prompt_id": prompt_row["prompt_id"],
                    "question_id": prompt_row["question_id"],
                    "prompt_style": prompt_row["prompt_style"],
                    "prompt": prompt_row["prompt"],
                }
            )

    out = pd.DataFrame(rows)

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {len(out)} mapped claim-prompt pairs to {output_path}")

    if len(out) > 0:
        print("\nMapped rows by claim category:")
        print(out["claim_category"].value_counts())

        print("\nUnique prompts by claim category:")
        print(out.groupby("claim_category")["prompt_id"].nunique())


if __name__ == "__main__":
    main()