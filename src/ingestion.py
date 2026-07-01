from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .models import JDFeatures
from .utils import normalize_whitespace, safe_float, safe_int

LOGGER = logging.getLogger(__name__)
W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def discover_dataset_root(start_dir: Path | None = None) -> Path:
    start = Path(start_dir or Path.cwd())
    candidates = list(start.rglob("candidates.jsonl"))
    if not candidates:
        raise FileNotFoundError("Could not locate candidates.jsonl under the workspace root.")
    preferred = [path.parent for path in candidates if path.parent.joinpath("candidate_schema.json").exists()]
    if preferred:
        return preferred[0]
    return candidates[0].parent


def load_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    texts = [node.text for node in root.findall(".//w:t", W_NS) if node.text]
    return normalize_whitespace(" ".join(texts))


def load_text_file(path: Path) -> str:
    return normalize_whitespace(path.read_text(encoding="utf-8"))


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_job_description(dataset_root: Path) -> str:
    for candidate in [
        dataset_root / "job_description.docx",
        dataset_root / "job_description.txt",
        dataset_root / "job_description.md",
        dataset_root / "jd.json",
    ]:
        if candidate.exists():
            if candidate.suffix.lower() == ".docx":
                return load_docx_text(candidate)
            if candidate.suffix.lower() == ".json":
                payload = load_json_file(candidate)
                if isinstance(payload, dict):
                    return normalize_whitespace(" ".join(str(value) for value in payload.values()))
                return normalize_whitespace(str(payload))
            return load_text_file(candidate)
    raise FileNotFoundError("No job description file found in the dataset directory.")


def load_candidate_schema(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "candidate_schema.json"
    return load_json_file(path)


def load_sample_candidates(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "sample_candidates.json"
    return load_json_file(path)


def load_sample_submission(dataset_root: Path) -> list[dict[str, str]]:
    path = dataset_root / "sample_submission.csv"
    if not path.exists():
        return []
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_jsonl_candidates(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_candidate_by_id(path: Path, candidate_id: str) -> dict[str, Any] | None:
    for candidate in load_jsonl_candidates(path):
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def build_candidate_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {}) or {}
    career_history = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    certifications = candidate.get("certifications", []) or []
    languages = candidate.get("languages", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    pieces: list[str] = []
    pieces.extend(
        [
            str(candidate.get("candidate_id", "")),
            str(profile.get("anonymized_name", "")),
            str(profile.get("headline", "")),
            str(profile.get("summary", "")),
            str(profile.get("current_title", "")),
            str(profile.get("current_company", "")),
            str(profile.get("current_industry", "")),
            str(profile.get("location", "")),
            str(profile.get("country", "")),
        ]
    )
    for entry in career_history:
        pieces.extend(
            [
                str(entry.get("title", "")),
                str(entry.get("company", "")),
                str(entry.get("industry", "")),
                str(entry.get("description", "")),
            ]
        )
    for entry in education:
        pieces.extend(
            [
                str(entry.get("institution", "")),
                str(entry.get("degree", "")),
                str(entry.get("field_of_study", "")),
                str(entry.get("grade", "")),
                str(entry.get("tier", "")),
            ]
        )
    for entry in skills:
        pieces.extend([str(entry.get("name", "")), str(entry.get("proficiency", ""))])
    for entry in certifications:
        pieces.extend([str(entry.get("name", "")), str(entry.get("issuer", ""))])
    for entry in languages:
        pieces.extend([str(entry.get("language", "")), str(entry.get("proficiency", ""))])

    pieces.extend(
        [
            str(signals.get("profile_completeness_score", "")),
            str(signals.get("open_to_work_flag", "")),
            str(signals.get("profile_views_received_30d", "")),
            str(signals.get("applications_submitted_30d", "")),
            str(signals.get("recruiter_response_rate", "")),
            str(signals.get("avg_response_time_hours", "")),
            str(signals.get("connection_count", "")),
            str(signals.get("endorsements_received", "")),
            str(signals.get("notice_period_days", "")),
            str(signals.get("preferred_work_mode", "")),
            str(signals.get("willing_to_relocate", "")),
            str(signals.get("github_activity_score", "")),
            str(signals.get("search_appearance_30d", "")),
            str(signals.get("saved_by_recruiters_30d", "")),
            str(signals.get("interview_completion_rate", "")),
            str(signals.get("offer_acceptance_rate", "")),
        ]
    )
    return normalize_whitespace(" \n".join(piece for piece in pieces if piece))


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate.get("profile", {}) or {}
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "candidate_name": profile.get("anonymized_name", ""),
        "current_title": profile.get("current_title", ""),
        "years_of_experience": safe_float(profile.get("years_of_experience", 0.0)),
    }


def parse_job_description_features(job_text: str) -> JDFeatures:
    normalized = normalize_whitespace(job_text)
    title = ""
    company = ""
    location = ""
    experience_min = None
    experience_max = None
    seniority = ""

    title_match = re.search(r"Job Description:\s*(.*?)\s*Company:", normalized, flags=re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    company_match = re.search(r"Company:\s*(.*?)\s*Location:", normalized, flags=re.IGNORECASE)
    if company_match:
        company = company_match.group(1).strip()
    location_match = re.search(r"Location:\s*(.*?)\s*Employment Type:", normalized, flags=re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip()
    experience_match = re.search(r"Experience Required:\s*([0-9]+(?:\.[0-9]+)?)\s*[–-]\s*([0-9]+(?:\.[0-9]+)?)\s*years", normalized, flags=re.IGNORECASE)
    if experience_match:
        experience_min = float(experience_match.group(1))
        experience_max = float(experience_match.group(2))
    else:
        experience_single = re.search(r"Experience Required:\s*([0-9]+(?:\.[0-9]+)?)\+?\s*years", normalized, flags=re.IGNORECASE)
        if experience_single:
            experience_min = float(experience_single.group(1))
    lowered = normalized.lower()
    if "founding team" in lowered or "senior" in lowered or "lead" in lowered:
        seniority = "senior"
    elif "staff" in lowered or "principal" in lowered:
        seniority = "staff"
    elif "manager" in lowered:
        seniority = "manager"

    keywords = [
        keyword
        for keyword in [
            "embeddings",
            "retrieval",
            "ranking",
            "search",
            "recommendation",
            "llm",
            "large language model",
            "fine-tuning",
            "fine tuning",
            "vector search",
            "rag",
            "prompt engineering",
            "evaluation",
            "python",
            "sql",
            "cloud",
            "mlops",
            "data pipelines",
            "product engineering",
        ]
        if keyword in lowered
    ]
    required_skills = []
    preferred_skills = []
    for keyword in keywords:
        if keyword in {"embeddings", "retrieval", "ranking", "llm", "fine-tuning", "vector search", "rag"}:
            required_skills.append(keyword)
        else:
            preferred_skills.append(keyword)

    responsibilities = []
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        sentence_lower = sentence.lower()
        if any(token in sentence_lower for token in ["what you'd actually be doing", "we need", "you'd be", "responsibilities", "build", "own"]):
            responsibilities.append(sentence)

    domain = "ai recruiting and ranking"
    if "search" in lowered or "retrieval" in lowered:
        domain = "ai search and ranking"

    return JDFeatures(
        raw_text=normalized,
        title=title,
        company=company,
        location=location,
        experience_min=experience_min,
        experience_max=experience_max,
        seniority=seniority,
        domain=domain,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities,
        keywords=keywords,
    )
