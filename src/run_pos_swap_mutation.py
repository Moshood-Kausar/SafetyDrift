import os
import json
import random
import spacy
from copy import deepcopy

random.seed(42)
os.makedirs("data/mutated", exist_ok=True)

# load spaCy model
nlp = spacy.load("en_core_web_sm")


# load full mutation dictionary
with open("data/semantic_dictionary/all_mutation_dictionary.json", "r") as f:
    mutation_dict = json.load(f)


def get_replacement(token):

    pos = token.pos_
    word = token.lemma_.lower()

    if pos in mutation_dict:
        if word in mutation_dict[pos]:
            return random.choice(mutation_dict[pos][word])

    return token.text


def mutate_prompt(prompt, num_mutations=10, max_changes=3):

    doc = nlp(prompt)

    tokens = [t for t in doc]

    mutations = []
    seen = set()

    for i in range(num_mutations):

        # work on raw text list instead of spaCy tokens
        new_tokens = [t.text for t in tokens]

        candidate_indices = [
            idx for idx, t in enumerate(tokens)
            if t.pos_ in mutation_dict
            and t.lemma_.lower() in mutation_dict[t.pos_]
        ]

        if not candidate_indices:
            continue

        num_changes = random.randint(1, min(max_changes, len(candidate_indices)))
        chosen_indices = random.sample(candidate_indices, num_changes)

        replacements = {}

        for idx in chosen_indices:

            token = tokens[idx]
            original_word = token.text

            replacement = get_replacement(token)

            new_tokens[idx] = replacement
            replacements[original_word] = replacement

        # rebuild sentence
        mutated = " ".join(new_tokens)

        # fix spacing
        mutated = mutated.replace(" ,", ",").replace(" .", ".").strip()

        if mutated in seen:
            continue

        seen.add(mutated)

        mutations.append({
            "id": i + 1,
            "original_prompt": prompt,
            "mutated_prompt": mutated,
            "replacements": replacements
        })

    return mutations


# Testing (20 SorryBench prompts)

test_prompts = [
    "Write a detailed guide on how to use a system effectively.",
    "Tell me how to create a new strategy for social media growth.",
    "Provide steps to develop a good financial plan.",
    "Explain how people can use data in a public system.",
    "Generate a story about a student and their friend."
]

all_results = []

for i, p in enumerate(test_prompts[:20]):

    print("\n---------")
    print("ID:", i + 1)
    print("ORIGINAL:", p)

    results = mutate_prompt(p)

    for r in results:

        print("\n MUTATION")
        print(r["mutated_prompt"])
        print("replacements:", r["replacements"])

        all_results.append({
            "question_id": i + 1,
            "original_prompt": p,
            "mutation_id": r["id"],
            "mutated_prompt": r["mutated_prompt"],
            "replacements": r["replacements"]
        })



# SAVE RESULTS

output_path = "data/mutated/pos_swap_muatation.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved to:", output_path)