import json
import os

from modifier_insertion import (
    modifier_mutation
)


INPUT_PATH = (
    "data/raw/"
    "sorrybench_base.json"
)

OUTPUT_PATH = (
    "data/mutated/"
    "sorrybench_modifier_mutation.json"
)

os.makedirs(
    "data/mutated",
    exist_ok=True
)


with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


all_results = []

# top 20 for testing
for item in data[:20]:

    question_id = item[
        "question_id"
    ]

    category = item[
        "category"
    ]

    prompt = item[
        "prompt"
    ]

    print("\n---------")
    print("ID:", question_id)
    print("ORIGINAL:", prompt)

    results = modifier_mutation(
        prompt
    )

    for r in results:

        print(
            f"\n[{r['modifier']}]"
        )
        print(
            r["mutated_prompt"]
        )

        all_results.append({
            "question_id":
                question_id,

            "category":
                category,

            "original_prompt":
                prompt,

            "mutation_type":
                "modifier_insertion",

            "modifier":
                r["modifier"],

            "mutated_prompt":
                r["mutated_prompt"]
        })


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    "\nSaved to:",
    OUTPUT_PATH
)