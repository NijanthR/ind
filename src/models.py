from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class JDFeatures:
    raw_text: str
    title: str = ""
    company: str = ""
    location: str = ""
    experience_min: float | None = None
    experience_max: float | None = None
    seniority: str = ""
    domain: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateSummary:
    candidate_id: str
    candidate_name: str
    current_title: str
    years_of_experience: float
    semantic_score: float = 0.0
    skills_match_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    platform_activity_score: float = 0.0
    career_trajectory_score: float = 0.0
    anomaly_penalty: float = 0.0
    base_score: float = 0.0
    final_score: float = 0.0
    recommendation_tier: str = ""
    reasoning: str = ""
    feature_breakdown: dict[str, Any] = field(default_factory=dict)
    raw_profile_text: str = ""
