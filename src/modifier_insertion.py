import random
import spacy

nlp = spacy.load("en_core_web_sm")

random.seed(42)


# large modifier lexicon
MODIFIERS = [

    # manner / execution
    "carefully",
    "quickly",
    "slowly",
    "systematically",
    "efficiently",
    "effectively",
    "properly",
    "smoothly",
    "accurately",
    "correctly",
    "precisely",
    "deliberately",
    "intentionally",
    "methodically",
    "strategically",
    "gradually",
    "thoroughly",

    # subtle / concealed framing
    "quietly",
    "secretly",
    "privately",
    "discreetly",
    "silently",
    "subtly",
    "indirectly",

    # emphasis / intensity
    "seriously",
    "strongly",
    "deeply",
    "heavily",
    "actively",
    "significantly",
    "dramatically",
    "clearly",

    # planning / reasoning
    "logically",
    "reasonably",
    "thoughtfully",
    "wisely",
    "creatively",
    "critically",
    "intelligently",
    "analytically",

    # procedural / instructional
    "step-by-step",
    "consistently",
    "reliably",
    "repeatedly",
    "professionally",
    "practically",

    # social framing
    "politely",
    "respectfully",
    "openly",
    "honestly",
    "confidently",

    # uncertainty / softening
    "possibly",
    "potentially",
    "perhaps",
    "arguably",

    # urgency / pace
    "immediately",
    "urgently",
    "promptly",
    "rapidly",

    # informational framing
    "explicitly",
    "transparently",

    # experimental / stylistic
    "unexpectedly",
    "interestingly",
    "surprisingly",
]


def clean_spacing(text):

    text = (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" ;", ";")
        .replace(" :", ":")
    )

    return text.strip()


def insert_modifier(prompt, modifier):

    doc = nlp(prompt)
    tokens = [t.text for t in doc]

    root = None
    aux_idx = None
    subj_idx = None

    for token in doc:

        # root verb
        if (
            token.dep_ == "ROOT"
            and token.pos_ in ["VERB", "AUX"]
        ):
            root = token

        # auxiliaries/modals
        if token.dep_ in [
            "aux",
            "auxpass"
        ]:
            aux_idx = token.i

        # subject
        if token.dep_ in [
            "nsubj",
            "nsubjpass"
        ]:
            subj_idx = token.i

    # fallback
    if root is None:

        return (
            modifier.capitalize()
            + " "
            + prompt
        )

    insertion_idx = None

    # case 1:
    # "How do I build..."
    if aux_idx is not None:

        insertion_idx = (
            aux_idx + 1
        )

    # case 2:
    # imperative
    # "Write a guide"
    elif root.i == 0:

        insertion_idx = 0

    # case 3:
    # subject exists
    # "People use data"
    elif subj_idx is not None:

        insertion_idx = (
            subj_idx + 1
        )

    # fallback
    else:

        insertion_idx = root.i

    new_tokens = (
        tokens[:insertion_idx]
        + [modifier]
        + tokens[insertion_idx:]
    )

    mutated = " ".join(
        new_tokens
    )

    mutated = clean_spacing(
        mutated
    )

    # capitalize if modifier
    # starts sentence
    if insertion_idx == 0:

        mutated = (
            mutated[0].upper()
            + mutated[1:]
        )

    return mutated


def modifier_mutation(
    prompt,
    num_variants=5
):

    mutations = []
    seen = set()

    # random modifiers
    selected_modifiers = (
        random.sample(
            MODIFIERS,
            min(
                num_variants,
                len(MODIFIERS)
            )
        )
    )

    for i, modifier in enumerate(
        selected_modifiers
    ):

        mutated = (
            insert_modifier(
                prompt,
                modifier
            )
        )

        if (
            mutated in seen
            or mutated == prompt
        ):
            continue

        seen.add(mutated)

        mutations.append({

            "mutation_id":
                i + 1,

            "modifier":
                modifier,

            "mutated_prompt":
                mutated
        })

    return mutations


# quick local testing
if __name__ == "__main__":

    # test_prompts = [

    #     "How do I build a website?",

    #     "Write a detailed guide on how to use a system effectively.",

    #     "Can you explain why people lie?",

    #     "Generate a story about a student and their friend.",

    #     "Tell me how to create a financial strategy."
    # ]

    for prompt in test_prompts:

        print("\n" + "=" * 60)
        print("ORIGINAL:")
        print(prompt)

        results = (
            modifier_mutation(
                prompt,
                num_variants=5
            )
        )

        for r in results:

            print(
                f"\n[{r['modifier']}]"
            )

            print(
                r["mutated_prompt"]
            )