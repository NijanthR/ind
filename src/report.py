from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from .models import CandidateSummary


def _tier_color(tier: str) -> str:
    return {
        "Strong Hire": "#166534",
        "Consider": "#b45309",
        "Borderline": "#b91c1c",
        "Not a Fit": "#374151",
    }.get(tier, "#374151")


def _chart_svg(candidate: CandidateSummary) -> str:
    import math

    values = [
        candidate.semantic_score * 100.0,
        candidate.skills_match_score * 10.0,
        candidate.experience_score * 10.0,
        candidate.platform_activity_score * 20.0,
        (candidate.education_score + candidate.career_trajectory_score) * 10.0,
    ]
    labels = ["Semantic", "Skills", "Experience", "Platform", "Education/Career"]
    size = 220
    radius = 78
    center = size / 2
    angles = [(-90 + index * 72) * math.pi / 180.0 for index in range(5)]

    def point(value: float, angle: float) -> tuple[float, float]:
        scale = radius * max(0.0, min(1.0, value / 100.0))
        return center + scale * math.cos(angle), center + scale * math.sin(angle)

    grid_points = []
    for level in [0.25, 0.5, 0.75, 1.0]:
        ring = []
        for angle in angles:
            x, y = point(level * 100.0, angle)
            ring.append(f"{x:.1f},{y:.1f}")
        grid_points.append(f'<polygon points="{" ".join(ring)}" fill="none" stroke="rgba(148,163,184,0.28)" stroke-width="1"/>')

    spokes = []
    for angle in angles:
        x, y = point(100.0, angle)
        spokes.append(f'<line x1="{center:.1f}" y1="{center:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="rgba(148,163,184,0.35)" stroke-width="1"/>')

    polygon_points = []
    for value, angle in zip(values, angles):
        x, y = point(value, angle)
        polygon_points.append(f"{x:.1f},{y:.1f}")

    labels_markup = []
    for label, angle in zip(labels, angles):
        x, y = point(100.0, angle)
        offset_x = 0 if abs(x - center) < 10 else (-16 if x < center else 16)
        offset_y = -8 if y < center else 12
        labels_markup.append(
            f'<text x="{x + offset_x:.1f}" y="{y + offset_y:.1f}" text-anchor="middle" fill="#475569" font-size="10">{html.escape(label)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" aria-label="radar chart">'
        f"{''.join(grid_points)}{''.join(spokes)}"
        f'<polygon points="{" ".join(polygon_points)}" fill="rgba(37,99,235,0.24)" stroke="#2563eb" stroke-width="2"/>'
        f"{''.join(labels_markup)}</svg>"
    )


def generate_report(output_path: Path, candidates: list[CandidateSummary]) -> None:
    top_candidates = candidates[:10]
    cards = []
    for index, candidate in enumerate(top_candidates, start=1):
        color = _tier_color(candidate.recommendation_tier)
        reasoning = html.escape(candidate.reasoning)
        cards.append(
            f'''
            <article class="card" style="--tier-color:{color}">
              <div class="card-top">
                <div>
                  <div class="rank">#{index} {html.escape(candidate.candidate_name or candidate.candidate_id)}</div>
                  <div class="meta">{html.escape(candidate.candidate_id)} | {html.escape(candidate.current_title)}</div>
                </div>
                <div class="score">{candidate.final_score:.1f}</div>
              </div>
              <div class="pill">{html.escape(candidate.recommendation_tier)}</div>
              <div class="chart">{_chart_svg(candidate)}</div>
              <div class="reasoning">{reasoning}</div>
              <div class="breakdown">Semantic {candidate.semantic_score:.2f} | Skills {candidate.skills_match_score:.1f}/10 | Experience {candidate.experience_score:.1f}/10 | Platform {candidate.platform_activity_score:.1f}/5 | Education {candidate.education_score:.1f}/5 | Career {candidate.career_trajectory_score:.1f}/5</div>
            </article>
            '''
        )

    payload = json.dumps([asdict(candidate) for candidate in top_candidates], ensure_ascii=False)
    best_score = top_candidates[0].final_score if top_candidates else 0.0
    best_name = top_candidates[0].candidate_name if top_candidates else "N/A"
    html_output = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Candidate Ranking Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: rgba(15, 23, 42, 0.78);
      --panel-border: rgba(148, 163, 184, 0.22);
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #60a5fa;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: var(--text); background: radial-gradient(circle at top left, #1d4ed8 0%, transparent 30%), linear-gradient(160deg, #020617, #0f172a 55%, #111827); }}
    .wrap {{ max-width: 1300px; margin: 0 auto; padding: 32px 20px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 38px; letter-spacing: -0.03em; }}
    .sub {{ color: var(--muted); max-width: 900px; line-height: 1.5; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
    .card {{ background: var(--panel); border: 1px solid var(--panel-border); border-radius: 20px; padding: 18px; box-shadow: 0 20px 50px rgba(2, 6, 23, 0.35); backdrop-filter: blur(10px); }}
    .card-top {{ display: flex; align-items: start; justify-content: space-between; gap: 12px; }}
    .rank {{ font-size: 18px; font-weight: 700; }}
    .meta, .breakdown, .reasoning, .llm {{ color: var(--muted); line-height: 1.5; }}
    .score {{ font-size: 32px; font-weight: 800; color: var(--accent); }}
    .pill {{ display: inline-flex; margin: 12px 0 10px; padding: 6px 10px; border-radius: 999px; color: white; background: var(--tier-color); font-size: 12px; font-weight: 700; letter-spacing: 0.02em; }}
    .chart {{ display: flex; justify-content: center; margin: 4px 0 10px; }}
    .reasoning {{ min-height: 56px; margin: 6px 0 10px; }}
    .breakdown {{ font-size: 12px; border-top: 1px solid rgba(148,163,184,0.18); padding-top: 10px; }}
    .hero {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin: 18px 0 26px; }}
    .stat {{ background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(148,163,184,0.16); border-radius: 16px; padding: 14px 16px; }}
    .stat .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.14em; }}
    .stat .value {{ font-size: 26px; font-weight: 800; margin-top: 6px; }}
    @media (max-width: 700px) {{ h1 {{ font-size: 30px; }} .wrap {{ padding: 24px 14px 32px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AI Candidate Ranking Report</h1>
    <div class="sub">Top 10 candidates ranked for the senior AI engineering role. The cards show the hybrid score, explanation, and a radar view of the main dimensions used by the ranker.</div>
    <div class="hero">
      <div class="stat"><div class="label">Candidates Scored</div><div class="value">{len(candidates):,}</div></div>
      <div class="stat"><div class="label">Top Candidate</div><div class="value">{html.escape(best_name)}</div></div>
      <div class="stat"><div class="label">Best Score</div><div class="value">{best_score:.1f}</div></div>
    </div>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
  <script>
    window.__REPORT_DATA__ = {payload};
  </script>
</body>
</html>
"""
    output_path.write_text(html_output, encoding="utf-8")
