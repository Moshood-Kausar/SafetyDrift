"""
SafetyDrift — Claim Extraction Pipeline (v4)
=============================================
Extracts safety-related claims from LLM system card PDFs.

Three target categories:
  1. Mental Health & Self-Harm
  2. Harmful / Violent Content
  3. Fairness & Bias

Key change in v4:
  Section matching now uses EXACT section titles instead of keywords.
  This eliminates all false matches from wrong chapters.

  NOTE: This is a temporary solution tied to these two specific PDFs.
  A more generalizable approach (e.g. semantic section matching or
  LLM-based section classification) should replace this in future work.

Pipeline:
  PDF → TOC → Exact Title Match → Page-Range Extraction
      → SpaCy 3-Step NLP → Feature Extraction → CSV Outputs

Install:
  pip install pymupdf spacy scikit-learn pandas
  python -m spacy download en_core_web_lg
"""

import re
import sys
import fitz
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — EXACT SECTION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY EXACT TITLES (v4 approach):
#   Previous versions used keyword matching against TOC titles.
#   This caused two persistent problems:
#     1. Over-matching: "health" matched §8.14 Healthcare (appendix)
#        when we only wanted §4.3 Mental health evaluations.
#     2. Under-matching: GPT-5 title filters were too strict,
#        leaving GPT-5 with almost no extracted claims.
#
#   Exact title matching eliminates both problems by pinning each
#   category to the precise TOC entries we confirmed from the actual
#   documents. No guessing, no false matches.
#
# HOW MATCHING WORKS:
#   We compare the lowercase-stripped TOC title against each exact
#   title in this dict (also lowercased). If they match, we include
#   that section. No partial matching, no keyword substrings.
#
# LIMITATION (acknowledged):
#   This only works for these two specific PDFs. A new system card
#   would require updating this dict manually. The future approach
#   should use semantic similarity or an LLM classifier to identify
#   relevant sections automatically regardless of title wording.
#
# ─────────────────────────────────────────────────────────────────────────────

EXACT_SECTIONS = {
    "claude": {
        "mental_health_self_harm": [
            "4.3.1 Suicide and self-harm",
            "4.3.2 Disordered eating",
        ],
        "harmful_violent_content": [
            "4.1 Harmful request evaluations",
            "4.1.1 Single-turn harmful request evaluation results",
            "4.1.2 Single-turn benign request evaluation results",
            "4.1.3 Multi-turn testing results",
            "4.1.4 Harmful request evaluations discussion",
        ],
        "fairness_bias": [
            "4.4.1 Political bias and even-handedness",
            "4.4.2 Bias Benchmark for Question Answering",
            "4.4.3 Election integrity",
        ],
    },
    "gpt5": {
        "mental_health_self_harm": [
            "Dynamic Mental Health Benchmarks with Adversarial User Simulations",
            "Health",
        ],
        "harmful_violent_content": [
            "Disallowed Content",
            "Evaluations with Challenging Prompts",
            "Evaluations with Representative Prompts",
            "From Hard Refusals to Safe-Completions",
        ],
        "fairness_bias": [
            "Fairness and Bias: BBQ Evaluation",
            "Bias Evaluation",
            "First Person Fairness Evaluation",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SPACY FEATURE LEXICONS  (paper Table I)
# ══════════════════════════════════════════════════════════════════════════════
#
# Used in extract_features() to assign one of three verb types.
# Expanded in v4 to cover linking verbs and reporting verbs that
# system cards commonly use as root verbs.
#
# external_behavioral : model does something observable
# internal_state      : model was designed/trained for something
# aggregate_stat      : model scored/achieved something measurable
# ─────────────────────────────────────────────────────────────────────────────

EXTERNAL_BEHAVIORAL_VERBS = {
    # Safety/refusal actions
    "refuse", "decline", "reject", "block", "filter", "flag",
    "detect", "prevent", "stop", "avoid", "warn", "redirect",
    # Output actions
    "output", "generate", "produce", "provide", "give", "return",
    "respond", "answer", "reply", "complete", "suggest", "recommend",
    # Compliance actions
    "follow", "comply", "adhere", "maintain", "enforce", "apply",
    "handle", "address", "support", "assist", "direct", "point",
    # Observable behavioral verbs used in system cards
    "contain", "include", "indicate", "represent", "reflect",
    "exhibit", "display", "show", "demonstrate", "retain",
    "position", "validate", "invite", "encourage",
}

INTERNAL_STATE_VERBS = {
    "prioritize", "intend", "aim", "design", "train", "learn",
    "understand", "believe", "consider", "ensure", "seek",
    "strive", "attempt", "try", "want", "need", "require",
    "expect", "plan", "commit", "dedicate", "focus", "target",
    "introduce", "develop", "build", "create",
}

AGGREGATE_STAT_VERBS = {
    "achieve", "score", "reach", "attain", "perform", "obtain",
    "record", "measure", "evaluate", "test", "assess", "improve",
    "reduce", "increase", "outperform", "exceed", "surpass",
    "match", "compare", "rate", "rank", "benchmark",
    # Reporting verbs used around benchmark numbers
    "report", "find", "observe", "note", "maintain", "publish",
}

STRONG_MODALS  = {"will", "must", "shall"}
DEONTIC_MODALS = {"should", "ought"}
POSS_MODALS    = {"may", "can", "might", "could"}

UNIVERSAL_QUANTIFIERS = {
    "always", "never", "all", "every", "none", "no",
    "entirely", "completely", "absolutely", "consistently",
}
HEDGED_QUANTIFIERS = {
    "may", "sometimes", "can", "might", "possibly", "often",
    "generally", "typically", "approximately", "around",
    "usually", "largely", "mostly", "tend", "likely", "unlikely",
}

MODEL_REFERENCE_WORDS = {
    "model", "system", "claude", "gpt", "assistant",
    "opus", "sonnet", "haiku", "gpt-5", "gpt-4", "chatgpt", "it",
}

# ── Noise patterns ────────────────────────────────────────────────────────────
# Sentences matching any of these are discarded before SpaCy processes them.
# Each pattern targets a known category of non-claim text from the PDFs.

NOISE_PATTERNS = [
    r'^\s*\d+\s*$',                        # lone page number
    r'^\s*table\s+\d+',                    # "Table 1", "Table 2"
    r'^\s*figure\s+\d+',                   # "Figure 1"
    r'^\s*\[?\d+\]',                       # citation [1]
    r'https?://',                           # URLs
    r'©|copyright|all rights reserved',    # copyright
    r'^\s*[A-Z\s]{10,}\s*$',              # ALL-CAPS headings
    r'higher is better|lower is better',   # table notes
    r'closer to zero is better',           # table notes
    r'arxiv',                               # arXiv refs
    r'^\s*\d+\.\d+\s+\d+\.\d+',           # bare number rows
    r'doi\.org|doi:',                       # DOI refs
    r'et al\.',                             # citation fragments
    r'ibid\.',                              # ibid
    r'bold.* indicat|indicat.* bold',      # table footnotes
    r'second.best score',                   # table footnotes
    r'does not take into account',          # table footnotes
    r'^\s*\w+\s+with\s+(thinking|system prompt)',  # eval config lines
    r'evaluations are run',                 # eval config lines
    r'rates are an average',                # table footnotes
    r'results.*system cards? due to',      # methodology disclaimers
    r'^\s*underlined',                      # table footnotes
    r'margin of error',                     # table footnotes
    r'^\s*[A-Za-z\s]+\(%\)',               # table column headers
    r'single-turn requests posing',         # table headers
    r'requests posing potential risk',      # table headers
    r'reflected in the scores below',       # setup sentences
    r'subject to some variation',           # disclaimer sentences
    r'values may vary slightly',            # disclaimer sentences
]


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — PDF LOADING AND TOC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(pdf_path: str) -> fitz.Document:
    """
    Opens a PDF using PyMuPDF and returns a Document object.

    fitz.open()      → reads the PDF and builds an in-memory model
    doc[page_idx]    → access one page (0-indexed internally)
    .get_text()      → extract all text from a page as a string
    .get_toc()       → read the embedded bookmark/outline structure

    INPUT  : file path string
    OUTPUT : fitz.Document
    """
    print(f"\n{'='*60}")
    print(f"  Loading: {pdf_path}")
    print(f"{'='*60}")
    doc = fitz.open(pdf_path)
    print(f"  Pages: {len(doc)}")
    return doc


def extract_toc(doc: fitz.Document) -> list:
    """
    Extracts the Table of Contents from the PDF.

    Returns a list of [level, title, page] entries where:
      level → nesting depth (1=top section, 2=subsection, etc.)
      title → section heading text exactly as written in the PDF
      page  → 1-based page number where the section starts

    PRIMARY:   doc.get_toc() reads embedded PDF bookmarks.
    FALLBACK:  parse_toc_visually() scans early pages with regex
               for PDFs without embedded bookmarks.

    INPUT  : fitz.Document
    OUTPUT : list of [level, title, page]
    """
    toc = doc.get_toc()
    if toc:
        print(f"  TOC: {len(toc)} entries (embedded)")
        return toc
    print("  No embedded TOC — scanning visually...")
    return parse_toc_visually(doc)


def parse_toc_visually(doc: fitz.Document, max_pages: int = 10) -> list:
    """
    Regex fallback TOC parser for PDFs without embedded bookmarks.

    Scans the first N pages for lines that look like TOC entries:
      section_number   title_text   ......   page_number

    REGEX:
      (\d+(?:\.\d+)*)  → section number like "4", "4.3", "4.3.1"
      ([A-Za-z]...)    → title (must start with letter)
      [.\s]{2,}        → TOC leader dots/spaces
      (\d{1,3})        → destination page

    LEVEL derived from dots in section number:
      "4"     → level 1
      "4.3"   → level 2
      "4.3.1" → level 3

    INPUT  : fitz.Document, max pages to scan
    OUTPUT : list of [level, title, page]
    """
    entries = []
    pattern = re.compile(
        r'^(\d+(?:\.\d+)*)\s+([A-Za-z][^\n]{4,70?}?)'
        r'\s*[.\s]{2,}\s*(\d{1,3})\s*$',
        re.MULTILINE
    )
    for idx in range(min(max_pages, len(doc))):
        for m in pattern.finditer(doc[idx].get_text()):
            num, title, page = m.groups()
            entries.append([len(num.split('.')), title.strip(), int(page)])
    print(f"  Visual TOC: {len(entries)} entries")
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — EXACT SECTION MAPPING (v4 core change)
# ══════════════════════════════════════════════════════════════════════════════

def build_section_map(toc: list, source_doc: str,
                      doc: fitz.Document) -> dict:
    """
    Maps each category to its matching TOC sections using EXACT titles.

    HOW IT WORKS:
      For each TOC entry [level, title, page]:
        1. Look up which category's exact_sections list contains
           this title (case-insensitive, strip whitespace).
        2. If found, record the section with its page range.
           end_page = start of the next TOC entry (or +10 for last).
        3. Tag the section type (evaluation / bullet / prose) to
           select the right extraction strategy later.

    SECTION TYPE TAGS:
      "evaluation" → sentences with numbers + performance verbs
      "bullet"     → each bullet point = one atomic claim
      "prose"      → full SpaCy 3-step NLP pipeline

    DEDUPLICATION:
      Tracks (start_page, end_page) pairs per category.
      Skips any section whose page range was already added
      to prevent double-extracting overlapping sections.

    WHY NOT KEYWORDS ANYMORE:
      Keywords were matching unrelated sections that happened to
      contain a common word. Exact title matching is deterministic —
      the same title always maps to the same category, no surprises.

    INPUT  : toc list, source doc name ("claude"/"gpt5"), fitz.Document
    OUTPUT : dict { category: [list of section dicts] }
    """
    section_map  = {cat: [] for cat in EXACT_SECTIONS[source_doc]}
    seen_ranges  = {cat: set() for cat in EXACT_SECTIONS[source_doc]}

    # Build lookup: lowercase title → (category, original title)
    lookup = {}
    for cat, titles in EXACT_SECTIONS[source_doc].items():
        for t in titles:
            lookup[t.lower().strip()] = cat

    for i, (level, title, page) in enumerate(toc):
        title_clean = title.lower().strip()
        if title_clean not in lookup:
            continue

        category = lookup[title_clean]
        end_page = toc[i + 1][2] if i + 1 < len(toc) else page + 10

        # Deduplicate on page range
        rng = (page, end_page)
        if rng in seen_ranges[category]:
            continue
        seen_ranges[category].add(rng)

        # Classify section type
        t_lower = title.lower()
        if any(w in t_lower for w in ["result", "evaluation",
                                       "benchmark", "score"]):
            sec_type = "evaluation"
        elif any(w in t_lower for w in ["risk", "limitation",
                                         "concern", "caveat"]):
            sec_type = "bullet"
        else:
            sec_type = "prose"

        section_map[category].append({
            "title":        title,
            "start_page":   page,
            "end_page":     end_page,
            "section_type": sec_type,
        })

    # Report matches
    print(f"\n  Section map [{source_doc.upper()}]:")
    for cat, secs in section_map.items():
        status = f"{len(secs)} sections" if secs else "NONE MATCHED"
        print(f"    {cat}: {status}")
        for s in secs:
            print(f"      └─ '{s['title']}' "
                  f"pp.{s['start_page']}–{s['end_page']} "
                  f"[{s['section_type']}]")

    return section_map


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — TEXT EXTRACTION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def get_section_text(doc: fitz.Document, section: dict) -> str:
    """
    Extracts and cleans text from a PDF page range.

    doc[page_idx].get_text():
      Returns all text on a page in reading order.
      Uses 0-based indexing; we convert from 1-based TOC pages.

    TEXT CLEANING:
      1. Replace newlines with spaces → fixes mid-sentence line breaks
         that PDF extraction preserves from the original layout.
      2. Collapse multiple spaces → cleans up any double-spacing.

    INPUT  : fitz.Document, section dict with start/end page
    OUTPUT : clean single-string text
    """
    raw = ""
    start = max(0, section["start_page"] - 1)   # 1-based → 0-based
    end   = min(len(doc), section["end_page"])
    for idx in range(start, end):
        raw += doc[idx].get_text()

    # Normalise whitespace: newlines → spaces, collapse runs
    clean = re.sub(r'\n+', ' ', raw)
    clean = re.sub(r' {2,}', ' ', clean)
    return clean.strip()


def is_noise(text: str) -> bool:
    """
    Returns True if text should be discarded (not a real claim).

    Checks against NOISE_PATTERNS — a list of regex patterns that
    match known non-claim text: table headers, table footnotes,
    figure captions, URLs, citations, methodology disclaimers, etc.

    INPUT  : text string
    OUTPUT : True = discard, False = may be a valid claim
    """
    for pat in NOISE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def extract_bullet_claims(text: str) -> list:
    """
    Extracts claims from bullet-point sections.

    FROM THE PAPER:
      Each bullet is already a standalone declarative statement,
      so no further segmentation is needed.

    BULLET MARKERS: •  -  *  –  or numbered (1. 2. 3.)
    MINIMUM LENGTH: 30 characters to filter empty/trivial bullets.

    INPUT  : cleaned section text
    OUTPUT : list of claim strings
    """
    claims  = []
    pattern = re.compile(
        r'(?:^|(?<=\s))(?:[•\-\*–]|\d+[\.\)])\s+(.{30,}?)(?='
        r'\s*(?:[•\-\*–]|\d+[\.\)])\s|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        claim = re.sub(r'\s+', ' ', m.group(1).strip())
        if claim and not is_noise(claim):
            claims.append(claim)
    return claims


def extract_benchmark_claims(text: str) -> list:
    """
    Extracts claims from evaluation/benchmark sections.

    FROM THE PAPER:
      Rows in the Evaluation section are parsed as benchmark claims
      of the form "The model achieves [score] on [benchmark]".

    A sentence is kept only if it has BOTH:
      1. A numeric value (%, decimal, or 2+ digit number)
      2. A performance verb (achieves, scores, outperforms, etc.)

    These are the most reliably testable claims in any system card
    because they reference a specific measurable outcome.

    INPUT  : cleaned section text
    OUTPUT : list of benchmark claim strings
    """
    claims   = []
    numeric  = re.compile(r'\d+\.?\d*\s*%|\b0\.\d+\b|\b\d{2,}\b')
    perf_pat = re.compile(
        r'\b(achieves?|scores?|performs?|reaches?|reduces?|improves?|'
        r'outperforms?|demonstrates?|reports?|records?|attains?|'
        r'exceeds?|surpasses?|shows?|maintains?|lowers?|finds?)\b',
        re.IGNORECASE
    )
    for sent in re.split(r'(?<=[.!?])\s+', text):
        sent = sent.strip()
        if len(sent) < 25 or is_noise(sent):
            continue
        if numeric.search(sent) and perf_pat.search(sent):
            claims.append(re.sub(r'\s+', ' ', sent))
    return claims


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — SPACY 3-STEP PIPELINE  (paper Section V-C1)
# ══════════════════════════════════════════════════════════════════════════════

def step1_segment_sentences(text: str, nlp) -> list:
    """
    STEP 1 — Sentence Segmentation

    Uses SpaCy's dependency-based segmenter, not period-splitting.

    WHY DEPENDENCY-BASED:
      Period-splitting incorrectly breaks at abbreviations ("Fig.")
      and decimal numbers ("98.5%"). SpaCy's parser understands
      grammar so it only splits at true sentence boundaries.

    SpaCy objects used:
      nlp(text)  → processes text, returns a Doc of Token objects
      doc.sents  → generator yielding Span objects (one per sentence)
      sent.text  → raw string of that sentence

    INPUT  : cleaned section text, loaded SpaCy model
    OUTPUT : list of SpaCy Span objects
    """
    return list(nlp(text).sents)


def step2_split_compounds(sentences: list) -> list:
    """
    STEP 2 — Compound Sentence Splitting

    Splits compound sentences into atomic units.

    WHY WE DO THIS:
      "The model refuses harmful requests and achieves 95% accuracy"
      → Two separate claims for RIPPER to classify individually.

    RULE:
      Split when a token has dep_="conj" (conjunct clause joined
      by "and/but/or") AND has its own subject (nsubj/nsubjpass).
      This identifies true independent clauses — not just any "and".

    SpaCy dependency labels:
      "ROOT"      → main verb
      "conj"      → conjunct (clause after "and/but/or")
      "nsubj"     → nominal subject
      "nsubjpass" → passive subject

    INPUT  : list of SpaCy Span objects
    OUTPUT : list of plain strings (may be longer than input)
    """
    results = []
    for sent in sentences:
        splits = [
            tok.i for tok in sent
            if tok.dep_ == "conj" and any(
                c.dep_ in ("nsubj", "nsubjpass") for c in tok.children
            )
        ]
        if not splits:
            results.append(sent.text.strip())
        else:
            cur = sent.start
            for sp in splits:
                chunk = sent.doc[cur:sp].text.strip()
                if chunk:
                    results.append(chunk)
                cur = sp
            tail = sent.doc[cur:sent.end].text.strip()
            if tail:
                results.append(tail)
    return results


def step3_filter(sentences: list, nlp) -> list:
    """
    STEP 3 — Noise Filtering

    Keeps only genuine declarative claim sentences.

    AN ATOMIC CLAIM is defined as:
      A self-contained declarative statement about a model's
      behavior, capability, limitation, or intended use that
      can be evaluated independently.

    DISCARD RULES (applied in order):
      1. Under 25 characters   → too short, likely a fragment
      2. Matches noise pattern  → table note, URL, citation, etc.
      3. No verb present        → heading or fragment, not a sentence
      4. Ends with "?"          → question, not a declarative claim

    SpaCy POS tags used:
      token.pos_ = "VERB" → main action verb
      token.pos_ = "AUX"  → helper verb (is/are/was/were)
      token.tag_ = "MD"   → modal verb (can/must/will/may/should)

    INPUT  : list of strings from step 2, SpaCy model
    OUTPUT : list of strings that passed all four rules
    """
    valid = []
    for text in sentences:
        text = text.strip()
        if len(text) < 25:
            continue
        if is_noise(text):
            continue
        has_verb = any(
            t.pos_ in ("VERB", "AUX") or t.tag_ == "MD"
            for t in nlp(text)
        )
        if not has_verb:
            continue
        if text.endswith('?'):
            continue
        valid.append(text)
    return valid


def run_spacy_pipeline(text: str, nlp) -> list:
    """
    Runs all three SpaCy steps in sequence on prose text.

    Called for sections tagged as "prose".
    Evaluation sections → extract_benchmark_claims()
    Bullet sections     → extract_bullet_claims()

    INPUT  : cleaned prose text, SpaCy model
    OUTPUT : list of atomic claim strings
    """
    sents    = step1_segment_sentences(text, nlp)
    atomic   = step2_split_compounds(sents)
    filtered = step3_filter(atomic, nlp)
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — FEATURE EXTRACTION  (paper Table I, 8 features, 4 groups)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(claim_text: str, nlp) -> dict:
    """
    Converts one claim sentence into the 8-feature vector (paper Table I).

    GROUP 1 — Term Frequency Vector
      Raw word counts across all claims. Computed separately in
      build_tf_matrix() because it needs the full corpus vocabulary.

    GROUP 2 — Quantification & Specificity
      contains_numeric        : has %, decimal, or number → testable
      has_universal_quantifier: "always"/"never" → strong assertion
      has_hedged_quantifier   : "may"/"typically" → weak/untestable

    GROUP 3 — Predicate Semantics
      verb_type:
        external_behavioral → observable action → testable
        internal_state      → design intent    → untestable
        aggregate_stat      → benchmark result → testable
      modal_verb_type:
        strong   (must/will)  → definite commitment
        deontic  (should)     → recommended
        possible (may/might)  → uncertain → untestable signal

    GROUP 4 — Syntactic Structure
      model_is_subject: model is the grammatical agent → testable
      tense           : present = ongoing, past = historical

    SpaCy properties used:
      token.lemma_      → base form ("refuses" → "refuse")
      token.dep_        → grammatical role (ROOT, nsubj, conj)
      token.pos_        → coarse POS (VERB, AUX, NOUN)
      token.tag_        → fine POS (MD = modal)
      token.like_num    → True if looks like a number
      token.lower_      → lowercase surface form
      token.morph.get() → morphological features (Tense)
      token.lefts       → left-side dependency children
      token.subtree     → all descendants in the parse tree

    INPUT  : claim string, SpaCy model
    OUTPUT : dict of feature_name → value
    """
    doc    = nlp(claim_text)
    tokens = list(doc)
    lower  = {t.lower_ for t in tokens}

    # Group 2
    contains_numeric = (
        any(t.like_num for t in tokens) or
        bool(re.search(r'\d+\.?\d*\s*%|\b0\.\d+\b', claim_text))
    )
    has_universal = bool(lower & UNIVERSAL_QUANTIFIERS)
    has_hedged    = bool(lower & HEDGED_QUANTIFIERS)

    # Group 3
    root       = next((t for t in tokens if t.dep_ == "ROOT"), None)
    root_lemma = root.lemma_.lower() if root else ""

    if   root_lemma in EXTERNAL_BEHAVIORAL_VERBS: verb_type = "external_behavioral"
    elif root_lemma in INTERNAL_STATE_VERBS:       verb_type = "internal_state"
    elif root_lemma in AGGREGATE_STAT_VERBS:       verb_type = "aggregate_stat"
    else:                                           verb_type = "other"

    modals = {t.lower_ for t in tokens if t.tag_ == "MD"}
    if   modals & STRONG_MODALS:  modal_type = "strong"
    elif modals & DEONTIC_MODALS: modal_type = "deontic"
    elif modals & POSS_MODALS:    modal_type = "possible"
    else:                          modal_type = "none"

    # Group 4
    model_is_subject = False
    if root:
        for subj in (t for t in root.lefts
                     if t.dep_ in ("nsubj", "nsubjpass")):
            if {t.lower_ for t in subj.subtree} & MODEL_REFERENCE_WORDS:
                model_is_subject = True
                break

    tense = "unknown"
    if root:
        m = root.morph.get("Tense")
        if m:
            tense = "present" if m[0].lower() in ("pres", "present") else "past"
        else:
            aux = {t.lower_ for t in tokens if t.pos_ == "AUX"}
            if aux & {"is", "are", "does", "do"}:  tense = "present"
            elif aux & {"was", "were", "did"}:      tense = "past"

    return {
        "claim_text":               claim_text,
        "contains_numeric":         contains_numeric,
        "has_universal_quantifier": has_universal,
        "has_hedged_quantifier":    has_hedged,
        "verb_type":                verb_type,
        "modal_verb_type":          modal_type,
        "model_is_subject":         model_is_subject,
        "tense":                    tense,
    }


def build_tf_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds Term Frequency vectors (Group 1 feature) across all claims.

    FROM THE PAPER:
      "a term frequency vector over all claims, where each dimension
       maps to a token in the vocabulary, and its value is the raw
       count of how many times that token appears in the claim."

    WHY COMPUTED SEPARATELY:
      TF vectors need a shared vocabulary across ALL claims.
      CountVectorizer.fit_transform() builds that vocabulary from
      the whole corpus then counts occurrences per claim.

    min_df=2     → ignore words in only 1 claim (too rare for rules)
    max_features → cap vocabulary to keep RIPPER rules readable
    stop_words   → remove "the", "a", "is" etc.

    INPUT  : DataFrame with 'claim_text' column
    OUTPUT : DataFrame with extra tf_WORD columns appended
    """
    vec = CountVectorizer(min_df=2, max_features=300,
                          stop_words='english')
    mat = vec.fit_transform(df["claim_text"])
    tf  = pd.DataFrame(
        mat.toarray(),
        columns=[f"tf_{w}" for w in vec.get_feature_names_out()]
    )
    return pd.concat([df.reset_index(drop=True), tf], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def extract_from_section(doc, section, category,
                          source_doc, nlp) -> list:
    """
    Extracts and featurises claims from one section.

    STRATEGY SELECTION:
      section_type = "evaluation" → extract_benchmark_claims()
      section_type = "bullet"     → extract_bullet_claims()
      section_type = "prose"      → run_spacy_pipeline()

    Each claim is tagged with metadata:
      category    → which of our 3 categories
      section     → section title it came from
      source_doc  → "claude" or "gpt5"
      section_type→ how it was extracted
      label       → None (annotators fill this in)

    INPUT  : PDF doc, section dict, category, source, SpaCy model
    OUTPUT : list of dicts (one per claim)
    """
    text     = get_section_text(doc, section)
    sec_type = section["section_type"]

    if   sec_type == "evaluation": texts = extract_benchmark_claims(text)
    elif sec_type == "bullet":     texts = extract_bullet_claims(text)
    else:                           texts = run_spacy_pipeline(text, nlp)

    records = []
    for t in texts:
        t = re.sub(r'\s+', ' ', t.strip())
        if not t or len(t) < 25:
            continue
        feat = extract_features(t, nlp)
        feat.update({
            "category":     category,
            "section":      section["title"],
            "source_doc":   source_doc,
            "section_type": sec_type,
            "label":        None,
        })
        records.append(feat)
    return records


def extract_all_claims(pdf_path: str, source_doc: str) -> pd.DataFrame:
    """
    Runs the full pipeline for one PDF.

    ORDER:
      1. load_pdf()             → open the PDF
      2. extract_toc()          → get section list
      3. build_section_map()    → exact title matching
      4. extract_from_section() → claims + features
      5. drop_duplicates()      → remove duplicate claim texts
      6. build_tf_matrix()      → add TF columns (Group 1)

    INPUT  : PDF path, source label
    OUTPUT : DataFrame of claims with all features
    """
    nlp = spacy.load("en_core_web_trf")
    doc = load_pdf(pdf_path)
    toc = extract_toc(doc)
    if not toc:
        print("  ERROR: Could not parse TOC.")
        return pd.DataFrame()

    section_map = build_section_map(toc, source_doc, doc)

    all_records = []
    for category, sections in section_map.items():
        print(f"\n  [{category}]")
        for section in sections:
            recs = extract_from_section(
                doc, section, category, source_doc, nlp)
            print(f"    '{section['title']}' → {len(recs)} claims")
            all_records.extend(recs)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    before = len(df)
    df = df.drop_duplicates(subset="claim_text", keep="first")
    if before - len(df):
        print(f"\n  Removed {before - len(df)} duplicates")

    df = build_tf_matrix(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    """
    Prints a human-readable summary after extraction.

    WHAT EACH SECTION SHOWS:

    Cross-tab (source × category):
      Every cell should have > 5 claims.
      ⚠ LOW flags cells that need attention.

    Feature distributions:
      contains_numeric  → how many claims have measurable values
      model_is_subject  → how many have the model as the actor
      verb_type         → behavioral vs state vs stat breakdown
      tense             → present (testable now) vs past (historical)

    Sample claims:
      2 per category to spot-check text quality.

    INPUT  : combined DataFrame
    """
    print("\n" + "="*60)
    print("  EXTRACTION SUMMARY")
    print("="*60)
    print(f"  Total claims : {len(df)}")
    print()

    print("  Source breakdown:")
    for src, n in df["source_doc"].value_counts().items():
        print(f"    {src:8s} → {n}")
    print()

    print("  Category × source:")
    cross = df.groupby(["source_doc", "category"]).size().reset_index(
        name="n")
    for _, r in cross.iterrows():
        flag = "✓" if r["n"] > 5 else "⚠ LOW"
        print(f"    {r['source_doc']:8s} | "
              f"{r['category']:35s} | {r['n']:3d}  {flag}")
    print()

    print("  Features:")
    print(f"    contains_numeric   : {df['contains_numeric'].sum()} / {len(df)}")
    print(f"    model_is_subject   : {df['model_is_subject'].sum()} / {len(df)}")
    print("    verb_type:")
    for v, c in df["verb_type"].value_counts().items():
        print(f"      {v:25s} : {c}")
    print("    tense:")
    for t, c in df["tense"].value_counts().items():
        print(f"      {t:10s} : {c}")
    print()

    print("  Sample claims (2 per category per source):")
    for cat in df["category"].unique():
        print(f"\n  [{cat}]")
        for src in df["source_doc"].unique():
            sub = df[(df["category"]==cat) & (df["source_doc"]==src)]
            if sub.empty:
                print(f"    [{src}] — no claims")
                continue
            for _, row in sub.head(2).iterrows():
                txt = row["claim_text"]
                print(f"    [{src}] {txt[:105]}"
                      f"{'...' if len(txt)>105 else ''}")

    print("\n" + "="*60)
    print()
    print("  NOTE: Section matching in v4 uses exact hardcoded titles.")
    print("  This works for these two PDFs only. Future work should")
    print("  replace this with a generalizable section classifier.")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════════════
# PART 10 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    CLAUDE_PDF = sys.argv[1] if len(sys.argv) > 1 \
        else "Claude_Opus_4_8_System_Card.pdf"
    GPT5_PDF   = sys.argv[2] if len(sys.argv) > 2 \
        else "gpt_5_model_cards.pdf"

    claude_df = extract_all_claims(CLAUDE_PDF, source_doc="claude")
    gpt5_df   = extract_all_claims(GPT5_PDF,   source_doc="gpt5")
    combined  = pd.concat([claude_df, gpt5_df], ignore_index=True)

    print_summary(combined)

    # OUTPUT 1 — Full feature matrix → fed into RIPPER after annotation
    combined.to_csv("claims_features.csv", index=False)
    print(f"  Saved: claims_features.csv  "
          f"({len(combined)} rows, {len(combined.columns)} cols)")

    # OUTPUT 2 — Annotation sheet → sent to 4 human annotators
    # 'label' column is blank — annotators assign one of:
    #   Direct Single-Turn | Proxy Single-Turn | Multi-Turn |
    #   Untestable | Unknown
    anno_cols = [
        "claim_text", "category", "section", "source_doc",
        "contains_numeric", "has_universal_quantifier",
        "has_hedged_quantifier", "verb_type", "modal_verb_type",
        "model_is_subject", "tense", "label"
    ]
    anno = combined[[c for c in anno_cols if c in combined.columns]]
    anno.to_csv("claims_for_annotation.csv", index=False)
    print(f"  Saved: claims_for_annotation.csv  ({len(anno)} rows)")
    print()
    print("  Next: annotators fill in 'label' column with:")
    print("  Direct Single-Turn | Proxy Single-Turn | Multi-Turn |"
          " Untestable | Unknown")