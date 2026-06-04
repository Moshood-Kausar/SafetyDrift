import json
from pathlib import Path

from voice_mutation import voice_mutation


# paths
input_path = Path(
    "data/raw/sorrybench_base.json"
)

output_path = Path(
    "data/mutated/sorrybench_voice_mutation.json"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# load dataset
with open(
    input_path,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)


results = []


# first 20 for testing
for item in dataset[:20]:

    question_id = item.get(
        "question_id"
    )

    category = item.get(
        "category"
    )

    prompt = item.get(
        "prompt"
    )

    mutation = voice_mutation(
        prompt
    )

    result = {
        "question_id":
            question_id,

        "category":
            category,

        "original_prompt":
            prompt,

        "mutation_type":
            mutation[
                "mutation_type"
            ],

        "mutated_prompt":
            mutation[
                "mutated_prompt"
            ]
    }

    results.append(result)

    print("\n--------")
    print(f"ID: {question_id}")
    print(f"ORIGINAL: {prompt}")
    print(
        f"TYPE: "
        f"{mutation['mutation_type']}"
    )
    print(
        f"MUTATED: "
        f"{mutation['mutated_prompt']}"
    )


# save results
with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"\nsaved to: "
    f"{output_path}"
)