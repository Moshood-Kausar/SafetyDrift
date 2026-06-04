# step4_map_claims_to_sorrybench.py

import pandas as pd
from pathlib import Path


TESTABLE_LABELS = {
    "Direct Single Turn",
    "Proxy Single Turn",
    "Multi Turn",
    "direct_single_turn",
    "proxy_single_turn",
    "multi_turn",
    "Testable",
    "testable",
}


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def map_claims_to_sorrybench(
    annotated_claims_csv: str,
    sorrybench_csv: str,
    output_csv: str,
):
    claims_df = pd.read_csv(annotated_claims_csv)
    sorry_df = pd.read_csv(sorrybench_csv)

    required_claim_cols = {
        "id",
        "claim_text",
        "label_claim",
        "label_testable",
        "category",
        "source_doc",
    }

    required_sorry_cols = {
        "prompt_id",
        "category",
        "prompt",
    }

    missing_claim_cols = required_claim_cols - set(claims_df.columns)
    missing_sorry_cols = required_sorry_cols - set(sorry_df.columns)

    if missing_claim_cols:
        raise ValueError(f"Annotated claims file is missing columns: {missing_claim_cols}")

    if missing_sorry_cols:
        raise ValueError(f"SorryBench file is missing columns: {missing_sorry_cols}")

    # Normalize important columns
    claims_df["label_claim_norm"] = claims_df["label_claim"].apply(normalize_text)
    claims_df["label_testable_norm"] = claims_df["label_testable"].apply(normalize_text)
    claims_df["category_norm"] = claims_df["category"].apply(normalize_text)

    sorry_df["category_norm"] = sorry_df["category"].apply(normalize_text)

    # Keep only actual testable claims
    testable_claims = claims_df[
        (claims_df["label_claim_norm"] != "Not a claim")
        & (claims_df["label_testable_norm"].isin(TESTABLE_LABELS))
        & (claims_df["category_norm"] != "")
    ].copy()

    print(f"Total annotated rows: {len(claims_df)}")
    print(f"Testable mapped claims kept: {len(testable_claims)}")

    rows = []

    for _, claim_row in testable_claims.iterrows():
        claim_category = claim_row["category_norm"]

        matching_prompts = sorry_df[
            sorry_df["category_norm"] == claim_category
        ]

        if matching_prompts.empty:
            print(
                f"WARNING: No SorryBench prompts found for category "
                f"'{claim_category}' from claim id {claim_row['id']}"
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
                    "sorrybench_category": claim_category,
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt": prompt_row["prompt"],
                }
            )

    out_df = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"Wrote {len(out_df)} claim-prompt pairs to {output_csv}")


if __name__ == "__main__":
    map_claims_to_sorrybench(
        annotated_claims_csv="outputs/annotated_claims.csv",
        sorrybench_csv="data/sorrybench.csv",
        output_csv="outputs/mapped_claims_to_prompts.csv",
    )