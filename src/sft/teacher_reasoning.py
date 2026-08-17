"""Phase 2E2: Kimi teacher for grounded rationale (JSON), not XML protocol.

Contract:
  Teacher → structured {"reasoning": "..."}  (content only)
  Code    → semantic validate + quality score
  Builder → deterministic <think>...</think> wrap
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.eval.metrics import normalize_answer
from src.sft.prototype_builder import resolve_evidence_refs, whitespace_norm

PROMPT_VERSION = "grounded_rationale_json_v2"
PROMPT_VERSION_V3 = "grounded_rationale_json_v3_2to6"
PROMPT_VERSION_V4 = "grounded_rationale_json_v4_deepseek"
DEFAULT_TEACHER_MODEL = "deepseek-v4-flash"

# Structured-output schema (OpenAI-compatible json_schema / json_object).
REASONING_JSON_SCHEMA: Dict[str, Any] = {
    "name": "teacher_reasoning",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Short grounded rationale bridging the gold evidence to "
                    "the gold answer (2-4 sentences)."
                ),
            }
        },
        "required": ["reasoning"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You write short grounded rationales for multi-hop QA supervision.

You are given a question, gold supporting evidence, and the gold answer.
Your job is ONLY to write the reasoning bridge from evidence → answer.

Rules:
1. Use ONLY facts present in the supplied evidence (and the question).
2. Explicitly connect entities/relations across the evidence pieces.
3. The rationale must lead to the provided gold answer (include it naturally).
4. Do not invent facts, titles, people, places, or dates.
5. Do not mention training, gold labels, prompts, or that you are an AI.
6. Do not output XML/HTML tags of any kind.
7. Keep it concise: normally 2-4 sentences (about 30-100 words).
8. For comparison questions: name the compared attributes and the outcome.

Return a JSON object with exactly one field:
{"reasoning": "<your rationale text>"}
"""

SYSTEM_PROMPT_V3 = """You write short grounded rationales for multi-hop QA supervision.

You are given a question, gold supporting evidence, and the gold answer.
Think internally as needed. Your saved output is ONLY the final justification.

Rules:
1. Use ONLY facts present in the supplied evidence (and the question).
2. Explicitly connect all required supporting facts to the gold answer.
3. Include the gold answer naturally at the end of the rationale.
4. Do not invent facts, titles, people, places, or dates.
5. Do not mention training, gold labels, prompts, or that you are an AI.
6. Do not output XML/HTML tags of any kind.
7. Use the minimum sufficient explanation, typically 2-6 sentences.
8. For comparison questions: name the compared attributes and the outcome.

Return a JSON object with exactly one field:
{"reasoning": "<your rationale text>"}
"""

SYSTEM_PROMPT_V4 = """You are generating supervised reasoning data for
evidence-grounded multi-hop question answering.

You are given:
1. a question,
2. the gold supporting evidence,
3. the reference answer.

Reason internally as much as necessary.

Return ONLY a JSON object:
{"reasoning": "..."}

The reasoning must:
- explicitly connect the necessary supporting facts;
- use only information supported by the provided evidence;
- naturally derive the reference answer;
- resolve entity or comparison relationships when needed;
- contain no invented facts;
- not mention "gold", "reference answer", "AI", prompt instructions,
  hidden reasoning, or grading;
- contain no XML or Markdown;
- be concise: normally 2-6 sentences;
- use the minimum sufficient explanation.
"""

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']{2,}|[0-9]{3,4}")
_PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_META_PATTERNS = (
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bthe user asks\b",
    r"\bwe need (to )?(answer|ensure|inspect|identify)\b",
    r"\blet's (inspect|check|see|think)\b",
    r"\btraining (example|data|sample)\b",
    r"\bgold (answer|evidence|label|labels)\b",
    r"\breference answer\b",
    r"\bhidden (reasoning|chain|cot)\b",
    r"\bgrading\b",
    r"\bi (will|need to|should|must)\b",
    r"\bmy (task|goal|job) is\b",
    r"\bstep[- ]by[- ]step\b",
    r"\b<think\b",
    r"\b</think\b",
    r"\b<answer\b",
    r"\b<evidence\b",
)
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)

_CONNECTIVE_RE = re.compile(
    r"\b(therefore|thus|hence|because|since|so|which means|"
    r"this (implies|shows|indicates|connects)|"
    r"comparing|compared|whereas|while|both)\b",
    re.IGNORECASE,
)

_GLUE_STOP = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "therefore",
    "thus",
    "hence",
    "because",
    "since",
    "which",
    "where",
    "when",
    "who",
    "whom",
    "whose",
    "what",
    "first",
    "second",
    "third",
    "evidence",
    "fact",
    "facts",
    "states",
    "state",
    "identifies",
    "identify",
    "requested",
    "answer",
    "born",
    "birthplace",
    "director",
    "directed",
    "comparing",
    "comparison",
    "attribute",
    "attributes",
    "larger",
    "smaller",
    "earlier",
    "later",
    "older",
    "younger",
    "more",
    "less",
    "both",
    "among",
    "however",
    "although",
    "though",
    "while",
    "after",
    "before",
    "during",
    "without",
    "within",
    "through",
    "between",
    "under",
    "over",
    "into",
    "onto",
    "whether",
    "either",
    "neither",
    "unless",
    "until",
    "according",
    "based",
    "together",
    "using",
    "given",
    "between",
    "across",
    "using",
    "only",
    "provided",
    "supporting",
    "according",
    "indicates",
    "implies",
    "shows",
    "connects",
    "establishes",
    "mentions",
    "refers",
    "it",
    "its",
    "another",
    "other",
    "others",
    "also",
    "they",
    "them",
    "their",
    "theirs",
    "such",
}

# Sentence-initial closed-class words that _PROPER_RE false-fires on.
_FUNCTION_PROPER = _GLUE_STOP | {
    "one",
    "two",
    "three",
    "four",
    "five",
    "each",
    "every",
    "these",
    "those",
    "there",
    "then",
}

# Leading modifiers stripped before a proper-noun span is judged novel.
_LEAD_STRIP = _FUNCTION_PROPER | {
    "excluding",
    "including",
    "republican",
    "democratic",
    "for",
}

# Conservative closed morphology only (not a general fuzzy matcher).
_MORPH_PAIRS = (
    ("award", "awards"),
    ("america", "american"),
    ("korea", "korean"),
)

_MONTH_NAME_RE = (
    r"january|february|march|april|may|june|july|"
    r"august|september|october|november|december"
)

# Calendar / date-style labels (not entity hallucinations).
_CALENDAR_CLOSED = {
    "old style",
    "new style",
    "julian",
    "gregorian",
}

# Country / demonym / continent families. A flagged noun is allowed when
# another member of the same family already appears in question/evidence/gold.
_GEO_FAMILIES: Tuple[frozenset, ...] = (
    frozenset({"italy", "italian", "italians"}),
    frozenset({"argentina", "argentine", "argentinian", "argentineborn"}),
    frozenset({"brazil", "brasil", "brazilian"}),
    frozenset(
        {
            "britain",
            "british",
            "uk",
            "united kingdom",
            "unitedkingdom",
            "great britain",
            "greatbritain",
        }
    ),
    frozenset({"scotland", "scottish", "glasgow"}),
    frozenset({"france", "french"}),
    frozenset({"spain", "spanish", "balearic"}),
    frozenset({"germany", "german"}),
    frozenset({"austria", "austrian"}),
    frozenset({"ireland", "irish"}),
    frozenset({"england", "english"}),
    frozenset({"wales", "welsh"}),
    frozenset({"canada", "canadian", "vancouver"}),
    frozenset(
        {
            "united states",
            "unitedstates",
            "usa",
            "america",
            "american",
            "washington",
        }
    ),
    frozenset(
        {
            "europe",
            "european",
            "france",
            "french",
            "italy",
            "italian",
            "spain",
            "spanish",
            "scotland",
            "scottish",
            "glasgow",
            "britain",
            "british",
            "england",
            "english",
            "germany",
            "german",
            "austria",
            "austrian",
            "ireland",
            "irish",
            "balearic",
        }
    ),
    frozenset(
        {
            "north america",
            "northamerica",
            "north american",
            "northamerican",
            "america",
            "american",
            "united states",
            "unitedstates",
            "usa",
            "canada",
            "canadian",
            "washington",
            "vancouver",
            "mexico",
            "mexican",
        }
    ),
    frozenset({"south america", "southamerica", "argentina", "argentine", "brazil", "brasil"}),
    frozenset({"south korea", "southkorea", "south korean", "southkorean", "korea", "korean"}),
    frozenset({"nazi", "nazis", "germany", "german"}),
    frozenset({"unitarian", "unitarians", "universalist", "universalists"}),
)

_NICK_PAIRS = {
    "mike": "michael",
    "michael": "mike",
    "geoff": "geoffrey",
    "geoffrey": "geoff",
    "niki": "nikola",
    "nikola": "niki",
    "nick": "nicholas",
    "nicholas": "nick",
    "jack": "john",
    "john": "jack",
    "alex": "alexander",
    "alexander": "alex",
    "bob": "robert",
    "robert": "bob",
    "bill": "william",
    "william": "bill",
    "jim": "james",
    "james": "jim",
    "tom": "thomas",
    "thomas": "tom",
}

_YES_RE = re.compile(
    r"\b(yes|same|both|share[ds]?|match(?:es|ed|ing)?|understand|"
    r"equivalent|affirmative|one (is )?(natural|man[- ]made))\b"
)
_NO_RE = re.compile(r"\b(no|not|neither|different|do not|does not|did not)\b")

_COUNTRY_ANSWER_ALIASES = {
    "british": {"british", "britain", "uk", "united kingdom", "great britain"},
    "britain": {"british", "britain", "uk", "united kingdom", "great britain"},
    "uk": {"british", "britain", "uk", "united kingdom"},
    "united kingdom": {"british", "britain", "uk", "united kingdom"},
    "italian": {"italian", "italy"},
    "italy": {"italian", "italy"},
    "american": {"american", "united states", "usa", "us"},
    "argentine": {"argentine", "argentina", "argentineborn"},
    "argentineborn": {"argentine", "argentina", "argentineborn"},
}


def format_teacher_user_prompt(
    sample: Dict[str, Any],
    refs: Sequence[Dict[str, Any]],
    gold_answer: str,
) -> str:
    qtype = (
        (sample.get("metadata") or {}).get("type")
        or sample.get("type")
        or "bridge"
    )
    extra = ""
    if str(qtype).lower() == "comparison":
        extra = (
            "\nThis is a comparison question: identify the compared attributes "
            "and state the comparison outcome.\n"
        )
    ev_lines = []
    for ref in refs:
        ev_lines.append(
            f"[{ref['title']}, {whitespace_norm(ref['text'])}]"
        )
    return (
        f"Question\n{sample['question']}\n\n"
        f"Supporting Evidence:\n"
        + "\n".join(ev_lines)
        + "\n\n"
        f"Reference Answer:\n{gold_answer}\n"
        + extra
        + '\nReturn JSON only: {"reasoning": "..."}'
    )


def wrap_think(reasoning: str) -> str:
    """Deterministic ECA protocol wrap — never ask the LLM to do this."""
    body = (reasoning or "").strip()
    return f"<think>\n{body}\n</think>"


def parse_teacher_json(raw: str) -> Tuple[Optional[str], List[str]]:
    """Parse teacher content into reasoning text. Returns (reasoning, errors)."""
    errors: List[str] = []
    text = (raw or "").strip()
    if not text:
        return None, ["empty teacher content"]

    # Strip common fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    obj: Any = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # salvage first {...} object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                return None, [f"json_decode_error: {exc}"]
        else:
            return None, ["content is not a JSON object"]

    if not isinstance(obj, dict):
        return None, ["json root is not an object"]
    if "reasoning" not in obj:
        return None, ["missing field: reasoning"]
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str):
        return None, ["reasoning must be a string"]
    reasoning = reasoning.strip()
    if not reasoning:
        return None, ["reasoning is empty"]
    if errors:
        return reasoning, errors
    return reasoning, []


def _vocab(texts: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for t in texts:
        for w in _WORD_RE.findall(t or ""):
            low = w.lower()
            out.add(low)
            for part in low.split("-"):
                if len(part) >= 3:
                    out.add(part)
    return out


def _content_tokens(text: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")} - _GLUE_STOP


def _evidence_coverage(
    reasoning: str, refs: Sequence[Dict[str, Any]]
) -> Tuple[List[bool], List[float]]:
    """Per-evidence: whether rationale shares content tokens with that evidence."""
    r_toks = _content_tokens(reasoning)
    used: List[bool] = []
    ratios: List[float] = []
    for ref in refs:
        e_toks = _content_tokens(
            f"{ref.get('title', '')} {ref.get('text', '')}"
        )
        if not e_toks:
            used.append(False)
            ratios.append(0.0)
            continue
        overlap = r_toks & e_toks
        # Require at least 2 content tokens, or 1 strong overlap if evidence tiny.
        need = 2 if len(e_toks) >= 4 else 1
        ok = len(overlap) >= need
        used.append(ok)
        ratios.append(round(len(overlap) / max(len(e_toks), 1), 4))
    return used, ratios


def _fold_letters(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", folded.lower())


def _fuzzy_name_in(needle: str, hay: str, thresh: float = 0.84) -> bool:
    """True if folded needle is a near-substring of folded hay (transliteration)."""
    short = _fold_letters(needle)
    long = _fold_letters(hay)
    if not short or not long:
        return False
    if short in long or (len(long) >= 5 and long in short):
        return True
    if len(short) < 5:
        return False
    n = len(short)
    for i in range(0, max(len(long) - n + 1, 1)):
        if SequenceMatcher(None, short, long[i : i + n]).ratio() >= thresh:
            return True
    return False


def _strip_lead_tokens(toks: Sequence[str]) -> List[str]:
    out = [t for t in toks]
    while out and out[0] in _LEAD_STRIP:
        out.pop(0)
    return out


def _morph_attested(tok: str, allowed: Set[str]) -> bool:
    if tok in allowed:
        return True
    for left, right in _MORPH_PAIRS:
        if tok == left and right in allowed:
            return True
        if tok == right and left in allowed:
            return True
    return False


def _canonical_dates(gold_raw: str, gold_n: str) -> List[str]:
    """Recover a legal date from HotpotQA concatenations; no arbitrary substrings."""
    blob = f"{gold_raw or ''} {gold_n or ''}"
    found: List[str] = []
    glued = re.search(
        rf"(\d{{4}})?(?P<mon>{_MONTH_NAME_RE})(?P<day>\d{{1,2}}),?\s*(?P<year>\d{{4}})",
        blob,
        flags=re.I,
    )
    if glued:
        month = glued.group("mon")
        day = str(int(glued.group("day")))
        year = glued.group("year")
        found.append(normalize_answer(f"{month} {day} {year}"))
        found.append(normalize_answer(f"{day} {month} {year}"))
    extra_year = re.search(
        rf"(?P<day>\d{{1,2}})\s+(?P<mon>{_MONTH_NAME_RE})\s+(?P<year>\d{{4}})\d+",
        gold_n or "",
        flags=re.I,
    )
    if extra_year:
        found.append(
            normalize_answer(
                f"{extra_year.group('day')} {extra_year.group('mon')} {extra_year.group('year')}"
            )
        )
    return [d for d in found if d]


def _geo_supported(prop: str, blob: str, allowed: Set[str]) -> bool:
    p = (prop or "").lower()
    pkey = _fold_letters(prop)
    prop_keys = {_fold_letters(t) for t in p.split() if t} | {pkey}
    blob_l = blob or ""
    blob_keys = {_fold_letters(w) for w in blob_l.split()} | {_fold_letters(blob_l)}
    for fam in _GEO_FAMILIES:
        keys = {_fold_letters(x) for x in fam}
        if p in fam or (prop_keys & keys):
            if any(k and (k in blob_keys or k in _fold_letters(blob_l)) for k in keys):
                return True
            if any(x in allowed for x in fam):
                return True
    return False


def _person_core_tokens(gold_raw: str, gold_n: str) -> List[str]:
    """Drop Wikipedia-lead junk so 'Franco Zeffirelli, KBE...' → name tokens."""
    head = (gold_raw or gold_n or "").split(",")[0]
    head = re.sub(r"\([^)]*\)", " ", head)
    head = re.split(
        r"\b(?:better known|born|kbe|omri|grande ufficiale)\b",
        head,
        flags=re.I,
    )[0]
    head_n = normalize_answer(head)
    drop = {
        "a",
        "an",
        "the",
        "born",
        "kbe",
        "omri",
        "grande",
        "ufficiale",
        "known",
        "better",
    }
    return [t for t in head_n.split() if t not in drop]


def _gold_aliases(gold_raw: str, gold_n: str) -> List[str]:
    aliases = [gold_n] if gold_n else []
    for quoted in re.findall(r'"([^"]+)"', gold_raw or ""):
        qn = normalize_answer(quoted)
        if qn:
            aliases.append(qn)
    m = re.search(r"better known as ([^,(]+)", gold_raw or "", flags=re.I)
    if m:
        kn = normalize_answer(m.group(1))
        if kn:
            aliases.append(kn)
    core = " ".join(_person_core_tokens(gold_raw, gold_n))
    if core:
        aliases.append(core)
    return [a for a in aliases if a]


def _token_aliases(tok: str) -> Set[str]:
    out = {tok}
    nick = _NICK_PAIRS.get(tok)
    if nick:
        out.add(nick)
    return out


def _answer_mentions_gold(
    gold_n: str,
    think_n: str,
    evidence_n: str = "",
    *,
    gold_raw: str = "",
    think_raw: str = "",
    question_n: str = "",
) -> bool:
    """True if the rationale derives the gold (alias / yes-no / symbol aware)."""
    if not think_n and not (think_raw or "").strip():
        return False
    if gold_n and gold_n in think_n:
        return True

    raw_gold = (gold_raw or "").strip()
    if raw_gold and len(raw_gold) <= 3 and raw_gold in (think_raw or ""):
        return True

    for alias in _COUNTRY_ANSWER_ALIASES.get(gold_n, ()):
        if alias and alias in think_n:
            return True

    for date_n in _canonical_dates(gold_raw, gold_n):
        if date_n and date_n in think_n:
            return True

    gtoks = [t for t in (gold_n or "").split() if t not in {"a", "an", "the"}]
    tset_set = set(think_n.split())
    if gtoks and all(t in tset_set or t in think_n for t in gtoks):
        return True

    for alias in _gold_aliases(gold_raw, gold_n):
        if alias and alias in think_n:
            return True

    core = _person_core_tokens(gold_raw, gold_n)
    distinctive = [t for t in core if len(t) >= 4] or [
        t for t in gtoks if len(t) >= 4
    ]
    conc_m = re.search(r"\b(?:therefore|thus|hence|so|since)\b(.{0,240})$", think_n)
    conc = conc_m.group(1) if conc_m else " ".join(think_n.split()[-12:])
    if len(core) >= 2 or len(gtoks) >= 2:
        first = distinctive[0] if distinctive else ""
        last = distinctive[-1] if distinctive else ""
        first_ok = bool(first) and any(a in think_n for a in _token_aliases(first))
        last_in_conc = bool(last) and last in conc
        # Require the gold surname/alias in the concluding clause so
        # "Herbert is older" cannot pass a Capriati gold.
        if last_in_conc and (
            first_ok
            or last in (question_n or "")
            or (gold_n and gold_n in (evidence_n or ""))
        ):
            return True

    # Long Wikipedia / phrase gold: accept a distinctive 3-gram overlap.
    if len(gtoks) >= 6:
        for n in (4, 3):
            for i in range(0, len(gtoks) - n + 1):
                gram = " ".join(gtoks[i : i + n])
                if gram in think_n:
                    return True

    if gold_n in {"yes", "no", "same", "different"}:
        if gold_n in {"yes", "same"} and (
            _YES_RE.search(think_n)
            or (
                re.search(r"\bnatural\b", think_n)
                and re.search(r"\bman[- ]made\b", think_n)
            )
        ):
            return True
        if gold_n in {"no", "different"} and _NO_RE.search(think_n):
            return True

    # Long / transliterated gold vs the question's surface name (Al-Nayrizi).
    gold_looks_foreign = len(gold_n or "") >= 24 or any(
        ord(c) > 127 for c in (gold_raw or gold_n or "")
    )
    if question_n and gold_looks_foreign:
        for span in re.findall(r"\b[a-z][a-z0-9]{4,}\b", question_n):
            if span in _FUNCTION_PROPER:
                continue
            if span in think_n and _fuzzy_name_in(span, gold_n):
                return True
    return False


def score_teacher_reasoning(
    reasoning: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 20,
    max_words: int = 150,
) -> Dict[str, Any]:
    """5-point quality score + semantic flags (no XML checks)."""
    errors: List[str] = []
    text = (reasoning or "").strip()
    n_words = len(text.split()) if text else 0

    # 1) answer consistency
    gold_n = normalize_answer(gold_answer)
    think_n = normalize_answer(text)
    evidence_n = normalize_answer(
        " ".join(
            [question, gold_answer]
            + [str(r.get("title") or "") for r in refs]
            + [str(r.get("text") or "") for r in refs]
        )
    )
    question_n = normalize_answer(question)
    answer_consistent = _answer_mentions_gold(
        gold_n,
        think_n,
        evidence_n,
        gold_raw=gold_answer,
        think_raw=text,
        question_n=question_n,
    )
    if not answer_consistent:
        errors.append("gold answer not found in normalized reasoning")

    # 2) grounding (novel proper nouns)
    allowed = _vocab(
        [question, gold_answer]
        + [r.get("title", "") for r in refs]
        + [r.get("text", "") for r in refs]
    )
    allowed |= _GLUE_STOP
    blob = " ".join(
        whitespace_norm(x).lower()
        for x in [question, gold_answer]
        + [r.get("title", "") for r in refs]
        + [r.get("text", "") for r in refs]
    )
    novel_props: List[str] = []
    geo_soft: List[str] = []
    for prop in _PROPER_RE.findall(text):
        toks = _strip_lead_tokens([t.lower() for t in prop.split()])
        if not toks:
            continue
        if all(t in _FUNCTION_PROPER for t in toks):
            continue
        content = [t for t in toks if len(t) >= 3]
        if not content:
            continue
        remainder = " ".join(content)
        if remainder in blob or all(_morph_attested(t, allowed) for t in content):
            continue
        if remainder in _CALENDAR_CLOSED or prop.lower() in _CALENDAR_CLOSED:
            continue
        if _geo_supported(remainder, blob, allowed) or _geo_supported(prop, blob, allowed):
            geo_soft.append(prop)
            continue
        novel_props.append(prop)
    grounding_valid = len(novel_props) == 0
    if not grounding_valid:
        errors.append(f"ungrounded proper nouns: {novel_props[:5]}")

    # 3-4) evidence lexical coverage
    ev_used, ev_ratios = _evidence_coverage(text, refs)
    while len(ev_used) < 2:
        ev_used.append(False)
        ev_ratios.append(0.0)
    evidence1_used = bool(ev_used[0]) if refs else False
    evidence2_used = bool(ev_used[1]) if len(refs) > 1 else False
    soft_warnings: List[str] = []
    if geo_soft:
        soft_warnings.append(f"geo_inference: {geo_soft[:5]}")
    if len(refs) >= 2 and not (evidence1_used and evidence2_used):
        soft_warnings.append(
            f"evidence_coverage incomplete: used={ev_used[:2]} ratios={ev_ratios[:2]}"
        )

    # 5) length
    length_valid = min_words <= n_words <= max_words
    if not length_valid:
        soft_warnings.append(f"word_count={n_words} not in [{min_words},{max_words}]")

    # meta / protocol pollution
    meta_hit = _META_RE.search(text)
    meta_clean = meta_hit is None
    if not meta_clean:
        errors.append(f"meta_or_protocol_phrase: {meta_hit.group(0)!r}")

    bridge_ok = bool(_CONNECTIVE_RE.search(text)) or (
        evidence1_used and evidence2_used
    )
    if not bridge_ok:
        soft_warnings.append("weak bridge: no connective and incomplete evidence use")

    score = int(answer_consistent) + int(grounding_valid) + int(evidence1_used)
    score += int(evidence2_used) + int(length_valid)
    # soft penalties (do not remove points already counted; gate via accepted)
    return {
        "answer_consistent": answer_consistent,
        "grounding_valid": grounding_valid,
        "evidence1_used": evidence1_used,
        "evidence2_used": evidence2_used,
        "evidence_coverage": ev_used[: len(refs)],
        "evidence_overlap_ratios": ev_ratios[: len(refs)],
        "length_valid": length_valid,
        "meta_clean": meta_clean,
        "bridge_ok": bridge_ok,
        "n_words": n_words,
        "novel_proper_nouns": novel_props,
        "quality_score": score,
        "errors": errors,
        "soft_warnings": soft_warnings,
    }


def validate_teacher_reasoning(
    raw_output: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 20,
    max_words: int = 150,
    min_accept_score: int = 4,
) -> Dict[str, Any]:
    """Layer1 parse JSON → Layer2 semantic score. XML format is NOT a gate."""
    reasoning, parse_errors = parse_teacher_json(raw_output)
    parse_ok = reasoning is not None and not parse_errors

    if not parse_ok:
        return {
            "parse_ok": False,
            "format_valid": False,  # legacy alias: parseable structured output
            "answer_consistent": False,
            "grounding_valid": False,
            "evidence1_used": False,
            "evidence2_used": False,
            "length_valid": False,
            "meta_clean": True,
            "bridge_ok": False,
            "n_words": 0,
            "novel_proper_nouns": [],
            "quality_score": 0,
            "errors": parse_errors or ["parse_failed"],
            "soft_warnings": [],
            "hard_reject": True,
            "accepted": False,
            "think": None,
            "reasoning": None,
            "think_wrapped": None,
        }

    scored = score_teacher_reasoning(
        reasoning or "",
        gold_answer=gold_answer,
        question=question,
        refs=refs,
        min_words=min_words,
        max_words=max_words,
    )
    errors = list(scored["errors"])
    soft_warnings = list(scored.get("soft_warnings") or [])
    xml_hit = bool(
        re.search(
            r"</?(?:search|observation|evidence|think|answer|internal)\b",
            reasoning or "",
            re.I,
        )
    )
    if xml_hit:
        errors.append("xml_or_protocol_tag_in_reasoning")
    n_sent = len([p for p in re.split(r"[.!?]+", reasoning or "") if p.strip()])
    if n_sent == 1:
        soft_warnings.append("too_short: 1 sentence")
    elif 7 <= n_sent <= 8:
        soft_warnings.append(f"slightly_long: {n_sent} sentences")
    rambling = n_sent >= 9 or int(scored.get("n_words") or 0) > 250
    if rambling:
        errors.append(f"rambling: sentences={n_sent} words={scored.get('n_words')}")
    # Coverage / 2-6 sentence band are audit signals, not hard rejects.
    # Hard reject: missing rationale, ungrounded facts, cannot derive answer,
    # XML / prompt / gold-style leak, obvious rambling.
    hard_reject = not (
        scored["answer_consistent"]
        and scored["grounding_valid"]
        and scored["meta_clean"]
        and not xml_hit
        and not rambling
        and bool((reasoning or "").strip())
    )
    accepted = not hard_reject
    if scored["quality_score"] < min_accept_score:
        soft_warnings.append(
            f"quality_score={scored['quality_score']}<{min_accept_score}"
        )
    if len(refs) >= 2 and not (
        scored["evidence1_used"] and scored["evidence2_used"]
    ):
        soft_warnings.append(
            "coverage heuristic miss (comparison/negation may be a false negative)"
        )

    wrapped = wrap_think(reasoning or "")
    return {
        "parse_ok": True,
        "format_valid": True,  # structured parse succeeded; XML is code-owned
        "answer_consistent": scored["answer_consistent"],
        "grounding_valid": scored["grounding_valid"],
        "evidence1_used": scored["evidence1_used"],
        "evidence2_used": scored["evidence2_used"],
        "evidence_coverage": scored["evidence_coverage"],
        "evidence_overlap_ratios": scored["evidence_overlap_ratios"],
        "length_valid": scored["length_valid"],
        "meta_clean": scored["meta_clean"],
        "bridge_ok": scored["bridge_ok"],
        "n_words": scored["n_words"],
        "novel_proper_nouns": scored["novel_proper_nouns"],
        "quality_score": scored["quality_score"],
        "errors": errors,
        "soft_warnings": soft_warnings,
        "hard_reject": hard_reject,
        "accepted": accepted,
        "think": reasoning,  # bare body for coldstart builder
        "reasoning": reasoning,
        "think_wrapped": wrapped,
    }


# Back-compat alias used by older call sites / docs.
def validate_teacher_think(
    raw_output: str,
    *,
    gold_answer: str,
    question: str,
    refs: Sequence[Dict[str, Any]],
    min_words: int = 20,
    max_words: int = 150,
) -> Dict[str, Any]:
    """Deprecated name: validates JSON rationale (not XML <think>)."""
    return validate_teacher_reasoning(
        raw_output,
        gold_answer=gold_answer,
        question=question,
        refs=refs,
        min_words=min_words,
        max_words=max_words,
    )


def mine_hard_candidates(
    *,
    samples_by_id: Dict[str, Dict[str, Any]],
    direct: Dict[str, Dict[str, Any]],
    base_oracle: Dict[str, Dict[str, Any]],
    sft_oracle: Dict[str, Dict[str, Any]],
    seed: int = 42,
    n_persistent: int = 320,
    n_other: int = 80,
) -> Tuple[List[str], Dict[str, Any]]:
    """Prefer persistent C-like hard; fill remainder from other v0-oracle-wrong."""
    import random

    def d_ok(sid: str) -> bool:
        r = direct.get(sid) or {}
        return bool(r.get("direct_correct")) or float(r.get("exact_match") or 0) >= 1.0 - 1e-9

    def o_ok(table: Dict[str, Dict[str, Any]], sid: str) -> bool:
        r = table.get(sid) or {}
        em = r.get("exact_match")
        if em is None and isinstance(r.get("metrics"), dict):
            em = r["metrics"].get("exact_match")
        return float(em or 0) >= 1.0 - 1e-9

    persistent: List[str] = []
    other_hard: List[str] = []
    for sid, sample in samples_by_id.items():
        if sid not in sft_oracle or o_ok(sft_oracle, sid):
            continue
        try:
            refs = resolve_evidence_refs(sample)
        except Exception:
            continue
        if len(refs) < 2:
            continue
        if (not d_ok(sid)) and (not o_ok(base_oracle, sid)):
            persistent.append(sid)
        else:
            other_hard.append(sid)

    rng = random.Random(seed)
    rng.shuffle(persistent)
    rng.shuffle(other_hard)
    chosen = persistent[:n_persistent]
    need = max(0, n_persistent + n_other - len(chosen))
    chosen.extend(other_hard[:need])
    if len(chosen) < n_persistent + n_other:
        rest = [s for s in other_hard if s not in chosen]
        chosen.extend(rest[: (n_persistent + n_other - len(chosen))])

    stats = {
        "n_persistent_available": len(persistent),
        "n_other_hard_available": len(other_hard),
        "n_chosen": len(chosen),
        "n_chosen_persistent": sum(1 for s in chosen if s in set(persistent)),
        "n_chosen_other": sum(1 for s in chosen if s not in set(persistent)),
        "n_persistent_target": n_persistent,
        "n_other_target": n_other,
    }
    return chosen, stats


def oracle_em_map_from_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = r["sample_id"]
        em = float((r.get("metrics") or {}).get("exact_match", 0) or 0)
        out[sid] = {"sample_id": sid, "exact_match": em, "metrics": r.get("metrics")}
    return out
