# dictionary_fallback.py
#
# -----------------------------------------------------------------------------
# 1. BASIC UTILS
# 2. SECTION / CATEGORY / PRIORITY HEURISTICS
# 3. DYNAMIC FALLBACK EXTRACTION
# 4. LLM QUALITY CHECK
# 5. MERGE LLM + FALLBACK
# 6. HYBRID EXTRACTION OUTPUT
# -----------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# 1. BASIC UTILS
# -----------------------------------------------------------------------------

def normalize_space(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_key(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def looks_like_duplicate(a: str, b: str) -> bool:
    na = normalize_key(a)
    nb = normalize_key(b)

    if not na or not nb:
        return False

    if na == nb:
        return True

    if na in nb or nb in na:
        shorter = min(len(na), len(nb))
        longer = max(len(na), len(nb))
        if shorter >= 18 and shorter / max(longer, 1) >= 0.72:
            return True

    a_tokens = set(na.split())
    b_tokens = set(nb.split())
    if not a_tokens or not b_tokens:
        return False

    overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    return overlap >= 0.72

# -----------------------------------------------------------------------------
# 2. SECTION / CATEGORY / PRIORITY HEURISTICS
# -----------------------------------------------------------------------------

SECTION_HEADERS = {
                    "must": [
                            "requirements", "required", "must have", "what you bring", "your profile",
                            "qualification", "qualifications", "we expect", "pozadujeme", "poziadavky",
                            "musite mat", "vyzadujeme", "profil kandidata", "minimum qualifications"
                            ],
                    "nice": [
                            "nice to have", "preferred", "bonus", "plus", "advantage", "vyhodou je",
                            "ocenime", "preferred qualifications"
                            ],
                    }

HARD_SKILL_PATTERNS = [
                        r"\bpython\b", r"\btypescript\b", r"\bsql\b", r"\bjava\b", r"\bc\+\+\b",
                        r"\bmachine learning\b", r"\bml\b", r"\bllm\b", r"\brag\b", r"\bgenerative ai\b",
                        r"\bdeep learning\b", r"\bstatistics\b", r"\bdata science\b", r"\bpower bi\b",
                        r"\bexcel\b", r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bdocker\b", r"\bkubernetes\b",
                        r"\bci/cd\b", r"\bgit\b", r"\belasticsearch\b", r"\blogstash\b", r"\bgrafana\b",
                        r"\bkibana\b", r"\bapi\b", r"\brest\b", r"\bscikit-learn\b", r"\bxgboost\b",
                        r"\bpandas\b", r"\bnumpy\b", r"\bpolars\b"
                        ]

EDUCATION_PATTERNS = [
                        r"\bbachelor", r"\bmaster", r"\bdoctorate", r"\bphd", r"\bdegree\b",
                        r"\buniversity\b", r"\bcomputer science\b", r"\bdata science\b",
                        r"\bapplied mathematics\b", r"\bstatistics\b", r"\bphysics\b",
                        r"\bvysokoskolske\b", r"\bvzdelanie\b", r"\btitul\b", r"\bfaculty\b"
                    ]

LANGUAGE_PATTERNS = [
                    r"\benglish\b", r"\banglick", r"\bgerman\b", r"\bnemeck", r"\bfrench\b",
                    r"\bczech\b", r"\bslovak\b", r"\bb2\b", r"\bc1\b", r"\bupper intermediate\b",
                    r"\badvanced\b", r"\bfluent\b"
                    ]

LOCATION_PATTERNS = [
                    r"\bbratislava\b", r"\bslovakia\b", r"\bhybrid\b", r"\bon-site\b", r"\bremote\b",
                    r"\bhome office\b", r"\bmlynske nivy\b", r"\btwin city\b"
                    ]

SOFT_SKILL_PATTERNS = [
                        r"\bcollaborative\b", r"\bteam\b", r"\bcommunication\b", r"\bagile\b",
                        r"\borchestration\b", r"\bplanning\b", r"\bcuriosity\b", r"\bproblem solving\b",
                        r"\bmentor", r"\bleadership\b", r"\bscaled agile\b"
                        ]

MUST_CUES = [
    "must", "required", "need to", "we require", "mandatory",
    "vyzadujeme", "pozadujeme", "nutne", "musi mat"
]

NICE_CUES = [
                "nice to have", "preferred", "bonus", "plus", "advantage",
                "ocenime", "vyhodou je", "preferovane"
            ]

def detect_section(line: str) -> Optional[str]:
    low = line.lower()
    for sec, headers in SECTION_HEADERS.items():
        for h in headers:
            if h in low:
                return sec
    return None

def classify_category(line: str) -> str:
    low = line.lower()

    if any(re.search(p, low) for p in EDUCATION_PATTERNS):
        return "education"
    if any(re.search(p, low) for p in LANGUAGE_PATTERNS):
        return "language"
    if any(re.search(p, low) for p in LOCATION_PATTERNS):
        return "location"
    if any(re.search(p, low) for p in HARD_SKILL_PATTERNS):
        return "hard_skill"
    if any(re.search(p, low) for p in SOFT_SKILL_PATTERNS):
        return "soft_skill"
    if re.search(r"\b\d+\+?\s*(year|years|rok|roky)\b", low):
        return "experience"

    return "other"

def infer_priority(line: str, current_section: Optional[str]) -> str:
    low = line.lower()

    if any(cue in low for cue in MUST_CUES):
        return "must"
    if any(cue in low for cue in NICE_CUES):
        return "nice"
    if current_section in {"must", "nice"}:
        return current_section

    return "unknown"

def line_score(line: str, current_section: Optional[str]) -> float:
    low = line.lower()
    score = 0.0

    # Very short fragments are noise; very long lines are usually full-page
    # flattening from a job portal. Long-but-structured lines are handled by
    # split_job_ad_lines() before scoring.
    if len(line) < 8 or len(line) > 650:
        return 0.0

    if len(line) > 320 and not current_section:
        return 0.0

    if re.match(r"^[•\-\*\u2022]", line):
        score += 2.0

    if any(cue in low for cue in MUST_CUES):
        score += 3.0
    if any(cue in low for cue in NICE_CUES):
        score += 2.0

    if current_section == "must":
        score += 2.0
    elif current_section == "nice":
        score += 1.0

    if re.search(r"\b\d+\+?\s*(year|years|rok|roky)\b", low):
        score += 2.0

    if any(re.search(p, low) for p in HARD_SKILL_PATTERNS):
        score += 2.5
    if any(re.search(p, low) for p in EDUCATION_PATTERNS):
        score += 2.5
    if any(re.search(p, low) for p in LANGUAGE_PATTERNS):
        score += 2.0
    if any(re.search(p, low) for p in LOCATION_PATTERNS):
        score += 1.5
    if any(re.search(p, low) for p in SOFT_SKILL_PATTERNS):
        score += 1.5

    if ":" in line and len(line.split(":")[0]) < 45:
        score += 0.5

    return score


# -----------------------------------------------------------------------------
# 3. DYNAMIC FALLBACK EXTRACTION
# -----------------------------------------------------------------------------

def split_job_ad_lines(job_text: str) -> List[str]:
    """Return stable, requirement-friendly lines from URL/manual job text.

    The URL scraper now preserves newlines, but this function also protects us
    from already-flattened text, copied web pages, and bullet lists embedded in
    a single paragraph.
    """
    text = str(job_text or "").replace("\r", "\n")

    # Put common bullets and HTML-list leftovers on their own line.
    text = re.sub(r"[\t ]*[•●▪▫◦‣⁃\u2022][\t ]*", "\n- ", text)
    text = re.sub(r"[\t ]+(?=-\s+)", "\n", text)
    text = re.sub(
        r"\s+(?=(requirements|qualifications|minimum qualifications|preferred qualifications|nice to have|what you bring|your profile|pozadujeme|poziadavky|vyzadujeme|vyhodou je)\s*:)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    raw = []
    for part in text.split("\n"):
        part = normalize_space(part)
        if not part:
            continue

        # Some portals flatten into one huge line. Split at strong section cues
        # and sentence boundaries before known requirement terms.
        if len(part) > 450:
            part = re.sub(
                r"\s+(?=(requirements|required|must have|what you bring|your profile|qualification|qualifications|we expect|pozadujeme|poziadavky|vyzadujeme|nice to have|preferred|bonus|advantage)\b)",
                "\n",
                part,
                flags=re.IGNORECASE,
            )
            part = re.sub(r"(?<=[.;])\s+(?=[A-Z][a-z])", "\n", part)

        for sub in part.split("\n"):
            sub = normalize_space(sub)
            if sub:
                raw.append(sub)

    # If still nearly one-line, split on semicolons and long dash-separated
    # requirement clusters. This is intentionally conservative to avoid turning
    # normal prose into nonsense.
    expanded = []
    for line in raw:
        if len(line) > 320 and (";" in line or " - " in line):
            chunks = re.split(r"\s*;\s*|\s+-\s+", line)
            chunks = [normalize_space(x) for x in chunks if normalize_space(x)]
            if len(chunks) >= 2:
                expanded.extend(chunks)
                continue
        expanded.append(line)

    # De-duplicate while keeping order.
    result = []
    seen = set()
    for line in expanded:
        key = normalize_key(line)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(line)

    return result


def fallback_extract_requirements_from_text(job_text: str, max_requirements: int) -> Dict[str, Any]:
    raw_lines = split_job_ad_lines(job_text)
    text = "\n".join(raw_lines)

    candidates = []
    current_section = None

    for line in raw_lines:
        maybe_section = detect_section(line)
        header_keys = {normalize_key(h) for values in SECTION_HEADERS.values() for h in values}
        line_key = normalize_key(line.rstrip(":-"))
        has_requirement_signal = any(re.search(p, line.lower()) for p in (HARD_SKILL_PATTERNS + EDUCATION_PATTERNS + LANGUAGE_PATTERNS + LOCATION_PATTERNS + SOFT_SKILL_PATTERNS))
        pure_section_header = (
            line_key in header_keys
            or (len(line) <= 60 and line.rstrip().endswith(":") and not has_requirement_signal)
        )
        if maybe_section is not None and pure_section_header:
            current_section = maybe_section
            continue

        # If the section header and content are on one line, keep the content
        # after the colon/dash and score it in that section.
        section_for_line = current_section
        maybe_inline_section = detect_section(line)
        line_to_score = line
        if maybe_inline_section is not None:
            section_for_line = maybe_inline_section
            if ":" in line and len(line.split(":", 1)[0]) <= 60:
                line_to_score = normalize_space(line.split(":", 1)[1])

        score = line_score(line_to_score, section_for_line)
        if score < 2.5:
            continue

        clean_line = re.sub(r"^[•\-\*\u2022]\s*", "", line_to_score).strip()
        clean_line = normalize_space(clean_line)
        clean_line = re.split(
            r"\b(benefits|what we offer|we offer|ponukame|benefity)\s*:",
            clean_line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        # Avoid carrying section-only headings as requirements, but do not drop
        # real requirements just because they contain words like "advantage".
        clean_key = normalize_key(clean_line.rstrip(":-"))
        if (not clean_line) or clean_key in header_keys:
            continue

        candidates.append({
            "text": clean_line,
            "category": classify_category(clean_line),
            "priority": infer_priority(clean_line, section_for_line),
            "score_hint": score,
        })

    # Last-resort extraction for technology lists that stayed in prose.
    if not candidates:
        skill_hits = []
        for pattern in HARD_SKILL_PATTERNS + LANGUAGE_PATTERNS + EDUCATION_PATTERNS:
            for m in re.finditer(pattern, text.lower()):
                hit = normalize_space(text[max(0, m.start() - 50): m.end() + 70])
                if 8 <= len(hit) <= 220:
                    skill_hits.append(hit)

        for hit in skill_hits[: max_requirements * 2]:
            candidates.append({
                "text": hit,
                "category": classify_category(hit),
                "priority": infer_priority(hit, None),
                "score_hint": 2.5,
            })

    dedup = []
    for item in candidates:
        is_dup = False
        for existing in dedup:
            if looks_like_duplicate(item["text"], existing["text"]):
                is_dup = True
                if item["score_hint"] > existing["score_hint"]:
                    existing.update(item)
                break
        if not is_dup:
            dedup.append(item)

    dedup.sort(key=lambda x: x["score_hint"], reverse=True)

    picked = []
    for item in dedup[:max_requirements]:
        priority = item["priority"]
        weight = 5.0 if priority == "must" else 2.0 if priority == "nice" else 1.5

        picked.append({
            "id": f"R{len(picked) + 1}",
            "text": item["text"],
            "category": item["category"],
            "priority": priority,
            "weight": weight,
        })

    job_title = "unknown"
    for line in raw_lines[:30]:
        if 8 <= len(line) <= 120 and any(x in line.lower() for x in ["engineer", "scientist", "developer", "analyst", "manager", "specialist", "consultant", "architect"]):
            job_title = line
            break

    seniority = "unknown"
    low_text = text.lower()
    if "senior" in low_text:
        seniority = "senior"
    elif "junior" in low_text:
        seniority = "junior"
    elif "medior" in low_text or "mid-level" in low_text or "mid level" in low_text:
        seniority = "medior"

    return {
            "job_title": job_title,
            "seniority": seniority,
            "requirements": picked,
            "_source": "fallback_dynamic_rules",
            }


# -----------------------------------------------------------------------------
# 4. LLM QUALITY CHECK
# -----------------------------------------------------------------------------

def is_weak_llm_extraction(llm_data: Dict[str, Any], min_requirements: int = 4) -> bool:
    reqs = llm_data.get("requirements", [])
    if not isinstance(reqs, list):
        return True

    clean_reqs = [r for r in reqs if isinstance(r, dict) and normalize_space(str(r.get("text", "")))]
    if len(clean_reqs) < min_requirements:
        return True

    categories = {str(r.get("category", "other")) for r in clean_reqs}
    must_count = sum(1 for r in clean_reqs if str(r.get("priority", "")) == "must")

    avg_len = 0.0
    if clean_reqs:
        avg_len = sum(len(normalize_space(str(r.get("text", "")))) for r in clean_reqs) / len(clean_reqs)

    if avg_len < 12:
        return True

    if len(categories) <= 1 and len(clean_reqs) <= 5:
        return True

    if must_count == 0 and len(clean_reqs) <= 5:
        return True

    return False

# -----------------------------------------------------------------------------
# 5. MERGE LLM + FALLBACK
# -----------------------------------------------------------------------------

def merge_requirement_lists(
    llm_requirements: List[Dict[str, Any]],
    fallback_requirements: List[Dict[str, Any]],
    max_requirements: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    def add_items(items: List[Dict[str, Any]]) -> None:
        nonlocal merged
        for item in items:
            if not isinstance(item, dict):
                continue

            text = normalize_space(str(item.get("text", "")))
            if not text:
                continue

            prepared = {
                "id": str(item.get("id") or f"R{len(merged) + 1}"),
                "text": text,
                "category": str(item.get("category") or "other"),
                "priority": str(item.get("priority") or "unknown"),
                "weight": float(item.get("weight", 1.0)),
            }

            matched = False
            for existing in merged:
                if looks_like_duplicate(prepared["text"], existing["text"]):
                    matched = True

                    # ak fallback vie lepsie trafit category / priority, mierne dopln
                    if existing.get("category") in {"other", "unknown"} and prepared["category"] not in {"other", "unknown"}:
                        existing["category"] = prepared["category"]

                    if existing.get("priority") == "unknown" and prepared["priority"] != "unknown":
                        existing["priority"] = prepared["priority"]

                    existing["weight"] = max(float(existing.get("weight", 1.0)), prepared["weight"])
                    break

            if not matched:
                merged.append(prepared)

    # preferujeme najprv LLM, potom doplnime fallback
    add_items(llm_requirements)
    add_items(fallback_requirements)

    # re-id
    final_items = []
    for i, item in enumerate(merged[:max_requirements], start=1):
        item["id"] = f"R{i}"
        final_items.append(item)

    return final_items

# -----------------------------------------------------------------------------
# 6. HYBRID EXTRACTION OUTPUT
# -----------------------------------------------------------------------------

def build_hybrid_requirement_result(
    llm_data: Dict[str, Any],
    fallback_data: Dict[str, Any],
    max_requirements: int,
) -> Dict[str, Any]:
    llm_requirements = llm_data.get("requirements", []) if isinstance(llm_data, dict) else []
    fallback_requirements = fallback_data.get("requirements", []) if isinstance(fallback_data, dict) else []

    weak_llm = is_weak_llm_extraction(llm_data)

    if weak_llm:
        merged_requirements = merge_requirement_lists([], fallback_requirements, max_requirements)
        source = "fallback_dynamic_rules_only"
    else:
        merged_requirements = merge_requirement_lists(llm_requirements, fallback_requirements, max_requirements)
        source = "llm_json_plus_fallback_merge"

    job_title = llm_data.get("job_title", "unknown") if isinstance(llm_data, dict) else "unknown"
    seniority = llm_data.get("seniority", "unknown") if isinstance(llm_data, dict) else "unknown"

    if job_title in {"", "unknown"}:
        job_title = fallback_data.get("job_title", "unknown")
    if seniority in {"", "unknown"}:
        seniority = fallback_data.get("seniority", "unknown")

    return {
            "job_title": job_title,
            "seniority": seniority,
            "requirements": merged_requirements,
            "_source": source,
            "_meta": {
            "llm_count": len(llm_requirements) if isinstance(llm_requirements, list) else 0,
            "fallback_count": len(fallback_requirements) if isinstance(fallback_requirements, list) else 0,
            "weak_llm": weak_llm,
            "merged_count": len(merged_requirements),},
            }