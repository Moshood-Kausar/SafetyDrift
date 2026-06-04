import spacy
import pyinflect

nlp = spacy.load("en_core_web_sm")


VALID_ROOTS = {
    "write",
    "tell",
    "give",
    "show",
    "explain",
    "describe",
    "create",
    "generate",
    "make",
    "build",
    "design",
    "draft",
    "compose",
    "recommend",
    "suggest",
    "teach"
}


def is_passive(doc):

    for token in doc:
        if token.dep_ == "auxpass":
            return True

    return False


def split_context_instruction(prompt):
    """
    split prompt into:
    context + instruction
    """

    sentences = [
        s.strip()
        for s in prompt.split(".")
        if s.strip()
    ]

    trigger_words = [
        "write",
        "tell",
        "give",
        "explain",
        "describe",
        "create",
        "generate",
        "design",
        "draft",
        "compose",
        "recommend",
        "suggest",
        "teach"
    ]

    instruction_idx = None

    for i in reversed(range(len(sentences))):

        lowered = sentences[i].lower()

        if any(
            lowered.startswith(word)
            or f" {word} " in lowered
            for word in trigger_words
        ):
            instruction_idx = i
            break

    if instruction_idx is None:
        return "", prompt

    context = ". ".join(
        sentences[:instruction_idx]
    )

    instruction = (
        sentences[instruction_idx]
    )

    return context, instruction


def active_to_passive(doc):

    root = None
    obj = None

    for token in doc:

        if (
            token.dep_ == "ROOT"
            and token.pos_ == "VERB"
        ):
            root = token

        if token.dep_ in [
            "dobj",
            "obj"
        ]:
            obj = token

    if root is None or obj is None:
        return None

    if (
        root.lemma_.lower()
        not in VALID_ROOTS
    ):
        return None

    # avoid pronouns
    if obj.pos_ == "PRON":
        return None

    participle = root._.inflect(
        "VBN"
    )

    if participle is None:
        return None

    obj_phrase = doc[
        obj.left_edge.i:
        obj.right_edge.i + 1
    ].text

    aux = (
        "are"
        if obj.tag_ in [
            "NNS",
            "NNPS"
        ]
        else "is"
    )

    mutated = (
        f"How {aux} "
        f"{obj_phrase} "
        f"{participle.lower()}?"
    )

    return mutated


def passive_to_active(doc):

    subj = None
    root = None

    for token in doc:

        if token.dep_ == (
            "nsubjpass"
        ):
            subj = token

        if (
            token.dep_ == "ROOT"
            and token.pos_ == "VERB"
        ):
            root = token

    if subj is None or root is None:
        return None

    obj_phrase = doc[
        subj.left_edge.i:
        subj.right_edge.i + 1
    ].text

    verb = root.lemma_

    mutated = (
        f"How do people "
        f"{verb} "
        f"{obj_phrase}?"
    )

    return mutated


def semantic_preserved(
    original_doc,
    mutated_doc
):
    """
    preserve meaning
    """

    important_pos = {
        "NOUN",
        "PROPN",
        "VERB",
        "ADJ"
    }

    original_words = {
        token.lemma_.lower()
        for token in original_doc
        if (
            token.pos_
            in important_pos
            and not token.is_stop
            and token.is_alpha
        )
    }

    mutated_words = {
        token.lemma_.lower()
        for token in mutated_doc
        if (
            token.pos_
            in important_pos
            and not token.is_stop
            and token.is_alpha
        )
    }

    if len(original_words) == 0:
        return False

    overlap = len(
        original_words.intersection(
            mutated_words
        )
    )

    ratio = (
        overlap
        / len(original_words)
    )

    return ratio >= 0.6


def semantic_safe_rewrite(
    prompt
):
    """
    safe rewrite when
    conversion fails
    """

    doc = nlp(prompt)

    root = None

    for token in doc:

        if (
            token.dep_ == "ROOT"
            and token.pos_ == "VERB"
        ):
            root = token
            break

    if root is None:
        return prompt

    root_lemma = (
        root.lemma_.lower()
    )

    content = doc[
        root.i + 1:
    ].text.strip()

    # cleanup
    content = (
        content
        .replace(
            "to me", ""
        )
        .replace(
            "me", "", 1
        )
        .strip(" .!?")
    )

    # remove leading "to"
    if content.lower().startswith(
        "to "
    ):
        content = content[3:]

    if len(content) < 3:
        return prompt

    content = (
        content[0].upper()
        + content[1:]
    )

    rewrite_templates = {
        "write":
            "{content} should be written.",

        "tell":
            "{content} should be told.",

        "give":
            "{content} should be given.",

        "show":
            "{content} should be shown.",

        "explain":
            "{content} should be explained.",

        "describe":
            "{content} should be described.",

        "create":
            "{content} should be created.",

        "generate":
            "{content} should be generated.",

        "design":
            "{content} should be designed.",

        "draft":
            "{content} should be drafted.",

        "compose":
            "{content} should be composed.",

        "recommend":
            "{content} should be recommended.",

        "suggest":
            "{content} should be suggested.",

        "teach":
            "{content} should be taught."
    }

    if (
        root_lemma
        in rewrite_templates
    ):

        return (
            rewrite_templates[
                root_lemma
            ].format(
                content=content
            )
        )

    return prompt


def voice_mutation(prompt):

    context, target_sentence = (
        split_context_instruction(
            prompt
        )
    )

    doc = nlp(
        target_sentence
    )

    mutation_type = (
        "active_to_passive"
    )

    # true passive only
    if is_passive(doc):

        mutation_type = (
            "passive_to_active"
        )

        mutated = (
            passive_to_active(
                doc
            )
        )

    else:

        mutated = (
            active_to_passive(
                doc
            )
        )

    # semantic validation
    if mutated is not None:

        mutated_doc = nlp(
            mutated
        )

        if not semantic_preserved(
            doc,
            mutated_doc
        ):

            mutated = (
                semantic_safe_rewrite(
                    target_sentence
                )
            )

    else:

        mutated = (
            semantic_safe_rewrite(
                target_sentence
            )
        )

    # restore context
    if (
        context
        and context.strip()
    ):

        mutated = (
            context.strip()
            + ".\n\n"
            + mutated
        )

    return {
        "original":
            prompt,

        "mutation_type":
            mutation_type,

        "mutated_prompt":
            mutated
    }