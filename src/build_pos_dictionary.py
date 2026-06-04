import json
import spacy
from collections import Counter

# load spacy
nlp = spacy.load("en_core_web_sm")

# load prompts
with open(
    "data/raw/sorrybench_base.json",
    "r"
) as f:
    prompts = json.load(f)

# allowed POS
allowed_pos = {
    "NOUN",
    "VERB",
    "ADJ"
}

# store words
word_bank = {
    "NOUN": Counter(),
    "VERB": Counter(),
    "ADJ": Counter()
}

for item in prompts:

    text = item["prompt"]

    doc = nlp(text)

    for token in doc:

        if (
            token.pos_ in allowed_pos
            and token.is_alpha
            and not token.is_stop
        ):

            word_bank[
                token.pos_
            ][
                token.text.lower()
            ] += 1


output = {}

for pos in word_bank:

    output[pos] = dict(
        word_bank[pos]
        .most_common()
    )

# save
with open(
    "data/candidate_words.json",
    "w"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )

print("saved candidate words")

import pandas as pd

rows = []

for pos in word_bank:

    for word, freq in (
        word_bank[pos]
        .most_common()
    ):

        rows.append({
            "word": word,
            "pos": pos,
            "frequency": freq
        })

df = pd.DataFrame(rows)

df.to_csv(
    "data/part_of_speech_words.csv",
    index=False
)

print(
    "saved csv to "
    "data/part_of_speech_words.csv"
)