import json

# load files
with open("data/semantic_dictionary/noun_mutation_dictionary.json", "r") as f:
    noun_dict = json.load(f)

with open("data/semantic_dictionary/verb_mutation_dictionary.json", "r") as f:
    verb_dict = json.load(f)

with open("data/semantic_dictionary/adj_mutation_dictionary.json", "r") as f:
    adj_dict = json.load(f)


# merge into one structure
final_dict = {
    "NOUN": noun_dict.get("NOUN", {}),
    "VERB": verb_dict.get("VERB", {}),
    "ADJ": adj_dict.get("ADJ", {})
}


# save final dictionary
with open("data/semantic_dictionary/all_mutation_dictionary.json", "w") as f:
    json.dump(final_dict, f, indent=2)

print("Final mutation dictionary created successfully")