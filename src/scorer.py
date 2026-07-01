from __future__ import annotations

import logging
import math
import re
from typing import Any

from datetime import date

from .config import DEGREE_SCORES, FIELD_HINTS, FAMILY_MAP, SKILL_SYNONYMS, TITLE_SENIORITY
from .models import CandidateSummary, JDFeatures
from .utils import clamp, format_sentence, mean, parse_date, safe_float, safe_int, string_similarity

LOGGER = logging.getLogger(__name__)


def _matches_any(text: str, patterns: list[str]) -> float:
    text_lower = text.lower()
    best = 0.0
    for pattern in patterns:
        if pattern in text_lower:
            return 1.0
        best = max(best, string_similarity(pattern, text_lower))
    return best


def _normalize_skill_text(value: str) -> str:
    return re.sub(r"[^a-z0-9+ ]+", " ", value.lower()).strip()


def _skill_family(skill_name: str) -> str:
    normalized = _normalize_skill_text(skill_name)
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if normalized == canonical or any(synonym in normalized for synonym in synonyms):
            return FAMILY_MAP.get(canonical, canonical)
    if any(keyword in normalized for keyword in ["ml", "ai", "nlp", "llm", "retrieval", "search", "ranking"]):
        return "ai"
    if any(keyword in normalized for keyword in ["cloud", "docker", "kubernetes", "aws", "gcp"]):
        return "platform"
    if any(keyword in normalized for keyword in ["sql", "spark", "airflow", "dbt", "pipeline"]):
        return "engineering"
    return "general"


def _candidate_skill_terms(candidate: dict[str, Any], profile_text: str) -> list[str]:
    terms: list[str] = []
    for skill in candidate.get("skills", []) or []:
        name = str(skill.get("name", "")).strip()
        if name:
            terms.append(name)
    for certification in candidate.get("certifications", []) or []:
        name = str(certification.get("name", "")).strip()
        if name:
            terms.append(name)
    lower_text = profile_text.lower()
    for canonical, synonyms in SKILL_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in lower_text:
                terms.append(canonical.replace("_", " "))
                break
    return sorted({term.strip() for term in terms if term.strip()})


def _match_score(jd_term: str, candidate_terms: list[str], profile_text: str) -> float:
    jd_norm = _normalize_skill_text(jd_term)
    if not candidate_terms:
        return 0.0

    best = 0.0
    jd_family = _skill_family(jd_norm)
    for candidate_term in candidate_terms:
        candidate_norm = _normalize_skill_text(candidate_term)
        similarity = string_similarity(jd_norm, candidate_norm)
        if jd_norm in candidate_norm or candidate_norm in jd_norm:
            similarity = max(similarity, 1.0)
        if similarity > best:
            best = similarity
        if jd_family != "general" and _skill_family(candidate_norm) == jd_family:
            best = max(best, 0.65)
    if jd_norm in profile_text.lower():
        best = max(best, 0.95)
    return clamp(best, 0.0, 1.0)


def _experience_target(jd: JDFeatures) -> float:
    if jd.experience_min is None and jd.experience_max is None:
        return 7.0
    if jd.experience_min is None:
        return float(jd.experience_max or 7.0)
    if jd.experience_max is None:
        return float(jd.experience_min)
    return (float(jd.experience_min) + float(jd.experience_max)) / 2.0


def _title_seniority(title: str) -> int:
    lower = title.lower()
    score = 0
    for token, value in TITLE_SENIORITY.items():
        if token in lower:
            score = max(score, value)
    return score


def _relevant_title_bonus(candidate: dict[str, Any], jd: JDFeatures) -> float:
    text = " ".join(
        [
            str(candidate.get("profile", {}).get("current_title", "")),
            str(candidate.get("profile", {}).get("summary", "")),
            " ".join(str(item.get("title", "")) for item in candidate.get("career_history", []) or []),
        ]
    ).lower()
    ai_match = _matches_any(text, ["ai engineer", "machine learning engineer", "ml engineer", "data scientist", "llm", "retrieval", "search", "ranking", "recommendation"])
    product_match = _matches_any(text, ["backend", "platform", "product", "full stack", "infra"])
    return 1.2 * ai_match + 0.4 * product_match


def score_skills(candidate: dict[str, Any], jd: JDFeatures, profile_text: str) -> tuple[float, list[str]]:
    required = list(dict.fromkeys(jd.required_skills))
    preferred = [skill for skill in jd.preferred_skills if skill not in required]
    candidate_terms = _candidate_skill_terms(candidate, profile_text)

    weighted_total = 0.0
    weighted_max = 0.0
    matched_terms: list[str] = []

    for skill in required:
        score = _match_score(skill, candidate_terms, profile_text)
        weighted_total += 2.0 * score
        weighted_max += 2.0
        if score >= 0.5:
            matched_terms.append(skill)

    for skill in preferred:
        score = _match_score(skill, candidate_terms, profile_text)
        weighted_total += 1.0 * score
        weighted_max += 1.0
        if score >= 0.5:
            matched_terms.append(skill)

    for skill in candidate_terms:
        if any(keyword in skill.lower() for keyword in ["kaggle", "leetcode", "codeforces"]):
            weighted_total += 0.15
            weighted_max += 0.15

    overlap = weighted_total / weighted_max if weighted_max else 0.0
    score = clamp(overlap * 10.0, 0.0, 10.0)
    return score, sorted({term for term in matched_terms if term})


def score_experience(candidate: dict[str, Any], jd: JDFeatures, profile_text: str) -> float:
    profile = candidate.get("profile", {}) or {}
    years = safe_float(profile.get("years_of_experience", 0.0))
    target = _experience_target(jd)
    min_years = jd.experience_min if jd.experience_min is not None else max(0.0, target - 2.0)
    max_years = jd.experience_max if jd.experience_max is not None else target + 2.0
    span = max(1.0, (max_years - min_years) or 4.0)
    distance = abs(years - target)
    base = 10.0 * math.exp(-distance / max(1.5, span))

    title_bonus = _relevant_title_bonus(candidate, jd)
    title_bonus += min(0.8, _title_seniority(str(profile.get("current_title", ""))) / 5.0)
    domain_bonus = 0.0
    if any(keyword in profile_text.lower() for keyword in ["ml", "machine learning", "llm", "retrieval", "ranking", "search", "embeddings", "nlp"]):
        domain_bonus += 0.8
    if any(keyword in profile_text.lower() for keyword in ["product", "shipping", "experiment", "evaluation", "latency", "quality"]):
        domain_bonus += 0.4

    if years < 1.0:
        base *= 0.5
    if years > 15.0 and jd.experience_max and years > jd.experience_max + 10:
        base *= 0.75

    return clamp(base * 0.6 + title_bonus + domain_bonus, 0.0, 10.0)


def score_education(candidate: dict[str, Any], jd: JDFeatures) -> float:
    education = candidate.get("education", []) or []
    if not education:
        return 0.0

    best = 0.0
    for entry in education:
        degree = str(entry.get("degree", "")).lower()
        field = str(entry.get("field_of_study", "")).lower()
        institution_tier = str(entry.get("tier", "unknown")).lower()

        degree_score = 0.0
        for key, value in DEGREE_SCORES.items():
            if key != "other" and key in degree:
                degree_score = max(degree_score, value)
        if not degree_score:
            degree_score = DEGREE_SCORES["other"]

        field_score = 0.0
        if any(hint in field for hint in FIELD_HINTS):
            field_score = 1.0
        elif any(hint in field for hint in ["engineering", "science", "technology"]):
            field_score = 0.6

        tier_score = {
            "tier_1": 0.8,
            "tier_2": 0.6,
            "tier_3": 0.4,
            "tier_4": 0.2,
            "unknown": 0.1,
        }.get(institution_tier, 0.1)

        candidate_score = clamp(degree_score + field_score + tier_score, 0.0, 5.0)
        best = max(best, candidate_score)

    if jd.domain and any(token in jd.domain.lower() for token in ["ai", "search", "ranking"]):
        best += 0.2
    return clamp(best, 0.0, 5.0)


def score_platform_activity(candidate: dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals", {}) or {}
    completeness = clamp(safe_float(signals.get("profile_completeness_score", 0.0)) / 100.0 * 5.0, 0.0, 5.0)
    github = safe_float(signals.get("github_activity_score", 0.0))
    github_component = 0.0 if github < 0 else clamp(github / 20.0, 0.0, 5.0)
    response = clamp(safe_float(signals.get("recruiter_response_rate", 0.0)) * 5.0, 0.0, 5.0)
    interview = clamp(safe_float(signals.get("interview_completion_rate", 0.0)) * 5.0, 0.0, 5.0)
    endorsements = clamp(min(safe_int(signals.get("endorsements_received", 0)), 100) / 20.0, 0.0, 5.0)
    connections = clamp(min(safe_int(signals.get("connection_count", 0)), 1000) / 200.0, 0.0, 5.0)
    views = clamp(min(safe_int(signals.get("profile_views_received_30d", 0)), 1000) / 250.0, 0.0, 5.0)
    recency = 0.0
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_since_active = max(0, (date.today() - last_active).days)
        recency = clamp(max(0.0, 1.0 - days_since_active / 180.0) * 5.0, 0.0, 5.0)
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}
    assessment = 0.0
    if assessment_scores:
        assessment = clamp(mean(safe_float(value) for value in assessment_scores.values()) / 20.0, 0.0, 5.0)

    open_to_work = 5.0 if signals.get("open_to_work_flag") else 0.0
    penalty = 0.0
    if safe_float(signals.get("avg_response_time_hours", 0.0)) > 72.0:
        penalty += 0.35
    if safe_int(signals.get("notice_period_days", 0)) > 90:
        penalty += 0.25
    if signals.get("willing_to_relocate") is False and str(signals.get("preferred_work_mode", "")).lower() == "onsite":
        penalty += 0.2

    components = [
        (completeness, 0.8),
        (github_component, 1.2),
        (response, 1.0),
        (interview, 0.7),
        (endorsements, 0.6),
        (connections, 0.3),
        (views, 0.25),
        (assessment, 0.85),
        (open_to_work, 0.4),
        (recency, 0.35),
    ]
    weighted = sum(value * weight for value, weight in components)
    max_weight = sum(weight for _, weight in components)
    score = clamp(weighted / max_weight, 0.0, 5.0)
    return clamp(score - penalty, 0.0, 5.0)


def score_career_trajectory(candidate: dict[str, Any]) -> float:
    history = list(candidate.get("career_history", []) or [])
    if not history:
        return 0.0

    sorted_history = sorted(history, key=lambda item: str(item.get("start_date", "")))
    seniority_values = [_title_seniority(str(item.get("title", ""))) for item in sorted_history]
    progress = 0.0
    for previous, current in zip(seniority_values, seniority_values[1:]):
        if current > previous:
            progress += 0.8
        elif current == previous:
            progress += 0.2
        else:
            progress -= 0.25

    tenures = [safe_float(item.get("duration_months", 0.0)) for item in sorted_history]
    average_tenure = mean(tenures)
    stability = 0.0
    if average_tenure >= 18.0:
        stability += 1.5
    elif average_tenure >= 12.0:
        stability += 1.0
    elif average_tenure >= 6.0:
        stability += 0.4
    else:
        stability -= 0.8

    short_stints = sum(1 for tenure in tenures if tenure < 12.0)
    if short_stints >= 2:
        stability -= 0.8

    same_company = len({str(item.get("company", "")).lower() for item in history}) < len(history)
    if same_company:
        progress += 0.5

    current_title = str(candidate.get("profile", {}).get("current_title", "")).lower()
    if any(keyword in current_title for keyword in ["senior", "lead", "principal", "staff"]):
        progress += 0.4
    if any(keyword in current_title for keyword in ["manager", "director", "vp"]):
        progress += 0.2

    score = 2.0 + progress + stability
    return clamp(score, 0.0, 5.0)


def compute_anomaly_penalty(candidate: dict[str, Any]) -> float:
    penalty = 0.0
    history = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}
    years = safe_float(profile.get("years_of_experience", 0.0))
    total_history_years = sum(safe_float(item.get("duration_months", 0.0)) for item in history) / 12.0
    if total_history_years > 0 and abs(total_history_years - years) > 8.0:
        penalty += 0.7

    for item in history:
        start = parse_date(item.get("start_date"))
        end = parse_date(item.get("end_date")) if item.get("end_date") else None
        duration_months = safe_float(item.get("duration_months", 0.0))
        if start and end and end < start:
            penalty += 0.8
        if duration_months < 0:
            penalty += 0.8
        if start is None:
            penalty += 0.2

    education = candidate.get("education", []) or []
    for item in education:
        start_year = safe_int(item.get("start_year", 0))
        end_year = safe_int(item.get("end_year", 0))
        if start_year and end_year and end_year < start_year:
            penalty += 0.4
        if start_year and (start_year < 1970 or start_year > 2035):
            penalty += 0.4

    skills = candidate.get("skills", []) or []
    if len(skills) > 0 and len([skill for skill in skills if safe_float(skill.get("endorsements", 0)) > 50]) > len(skills) / 2:
        penalty += 0.2

    if years > 0 and any(keyword in str(profile.get("current_title", "")).lower() for keyword in ["intern", "junior"]) and years >= 8:
        penalty += 0.8

    return clamp(penalty, 0.0, 3.0)


def build_reasoning(candidate: dict[str, Any], jd: JDFeatures, matched_skills: list[str], components: dict[str, float]) -> str:
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    top_skills = ", ".join(matched_skills[:4]) if matched_skills else "adjacent technical experience"
    work_context = []
    if history:
        first_role = history[0]
        work_context.append(str(first_role.get("title", "")))
        if first_role.get("company"):
            work_context.append(str(first_role.get("company", "")))
    core_sentence = f"{profile.get('current_title', 'Candidate')} with {safe_float(profile.get('years_of_experience', 0.0)):.1f} years and fit around {top_skills}"
    if jd.experience_min is not None and jd.experience_max is not None:
        core_sentence += f" for a {jd.experience_min:.0f}-{jd.experience_max:.0f} year AI engineering role"
    support_parts = [
        f"semantic {components.get('semantic_score', 0.0):.2f}",
        f"skills {components.get('skills_match_score', 0.0):.1f}/10",
        f"experience {components.get('experience_score', 0.0):.1f}/10",
    ]
    if work_context:
        support_parts.append("career path " + " -> ".join(work_context[:2]))
    return format_sentence([core_sentence + ".", "; ".join(support_parts)])


def recommendation_tier(score: float) -> str:
    if score >= 75.0:
        return "Strong Hire"
    if score >= 50.0:
        return "Consider"
    if score >= 30.0:
        return "Borderline"
    return "Not a Fit"


def score_candidate(candidate: dict[str, Any], jd: JDFeatures, semantic_score: float) -> CandidateSummary:
    profile = candidate.get("profile", {}) or {}
    profile_text = str(candidate.get("_profile_text", "")) or " ".join(
        [
            str(profile.get("summary", "")),
            str(profile.get("headline", "")),
            str(profile.get("current_title", "")),
        ]
    )
    semantic_score = clamp(semantic_score, 0.0, 1.0)
    skills_score, matched_skills = score_skills(candidate, jd, profile_text)
    experience_score = score_experience(candidate, jd, profile_text)
    education_score = score_education(candidate, jd)
    platform_score = score_platform_activity(candidate)
    career_score = score_career_trajectory(candidate)
    anomaly_penalty = compute_anomaly_penalty(candidate)

    base_score = 100.0 * (
        0.35 * semantic_score
        + 0.25 * (skills_score / 10.0)
        + 0.20 * (experience_score / 10.0)
        + 0.10 * (platform_score / 5.0)
        + 0.10 * ((education_score + career_score) / 10.0)
    )
    final_score = clamp(base_score - anomaly_penalty * 6.0, 0.0, 100.0)

    candidate_summary = CandidateSummary(
        candidate_id=str(candidate.get("candidate_id", "")),
        candidate_name=str(profile.get("anonymized_name", "")),
        current_title=str(profile.get("current_title", "")),
        years_of_experience=safe_float(profile.get("years_of_experience", 0.0)),
        semantic_score=semantic_score,
        skills_match_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        platform_activity_score=platform_score,
        career_trajectory_score=career_score,
        anomaly_penalty=anomaly_penalty,
        base_score=base_score,
        final_score=final_score,
        recommendation_tier=recommendation_tier(final_score),
        reasoning="",
        feature_breakdown={
            "matched_skills": matched_skills,
            "semantic_score": semantic_score,
            "skills_match_score": skills_score,
            "experience_score": experience_score,
            "education_score": education_score,
            "platform_activity_score": platform_score,
            "career_trajectory_score": career_score,
            "anomaly_penalty": anomaly_penalty,
            "base_score": base_score,
        },
        raw_profile_text=profile_text,
    )
    candidate_summary.reasoning = build_reasoning(candidate, jd, matched_skills, candidate_summary.feature_breakdown)
    return candidate_summary


def cheap_prefilter_score(candidate: dict[str, Any], jd: JDFeatures) -> float:
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    headline = str(profile.get("headline", "")).lower()
    summary = str(profile.get("summary", "")).lower()
    current_title = str(profile.get("current_title", "")).lower()
    current_industry = str(profile.get("current_industry", "")).lower()
    combined_text = f"{headline} {summary} {current_title} {current_industry}"
    title_match = _matches_any(combined_text, [jd.title.lower(), jd.domain.lower(), "ai engineer", "ml engineer", "machine learning engineer", "data scientist", "search", "ranking", "retrieval"])
    skill_match = 0.0
    for token in ["python", "sql", "spark", "airflow", "llm", "embeddings", "retrieval", "ranking", "rag", "fine-tuning", "vector search", "nlp", "mlops", "bentoml"]:
        if token in combined_text:
            skill_match += 0.06
    skill_match = clamp(skill_match, 0.0, 1.0)
    experience = safe_float(profile.get("years_of_experience", 0.0))
    exp_target = _experience_target(jd)
    exp_alignment = 1.0 - min(1.0, abs(experience - exp_target) / max(1.0, exp_target + 1.0))
    activity = safe_float(signals.get("profile_completeness_score", 0.0)) / 100.0
    github = safe_float(signals.get("github_activity_score", 0.0))
    github_bonus = 0.0 if github < 0 else github / 100.0
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    return 0.34 * title_match + 0.28 * (skill_match / 10.0) + 0.18 * exp_alignment + 0.12 * activity + 0.05 * github_bonus + 0.03 * open_to_work


def normalize_and_sort(candidates: list[CandidateSummary]) -> list[CandidateSummary]:
    ordered = sorted(candidates, key=lambda item: (-item.final_score, item.candidate_id))
    previous_score = 100.0
    for item in ordered:
        item.final_score = clamp(min(item.final_score, previous_score), 0.0, 100.0)
        item.recommendation_tier = recommendation_tier(item.final_score)
        previous_score = item.final_score
    return ordered


def debug_summary(candidate: CandidateSummary) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_name": candidate.candidate_name,
        "score": round(candidate.final_score, 4),
        "tier": candidate.recommendation_tier,
        "features": candidate.feature_breakdown,
    }
