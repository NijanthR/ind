from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path

import gradio as gr

if __package__ is None or __package__ == "":
    from src.embedder import SemanticEmbedder
    from src.ingestion import discover_dataset_root
    from src.ingestion import build_candidate_text, load_jsonl_candidates, load_job_description, parse_job_description_features
    from src.scorer import cheap_prefilter_score, normalize_and_sort, score_candidate
else:
    from .embedder import SemanticEmbedder
    from .ingestion import discover_dataset_root
    from .ingestion import build_candidate_text, load_jsonl_candidates, load_job_description, parse_job_description_features
    from .scorer import cheap_prefilter_score, normalize_and_sort, score_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str, base_dir: Path) -> Path | None:
    text = value.strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base_dir / path


def _format_preview_rows(scored_candidates: list, limit: int = 10) -> list[list[str]]:
    rows: list[list[str]] = []
    for rank, candidate in enumerate(scored_candidates[:limit], start=1):
        rows.append(
            [
                str(rank),
                str(candidate.candidate_id),
                str(candidate.candidate_name),
                str(candidate.current_title),
                f"{candidate.final_score:.2f}",
                str(candidate.recommendation_tier),
                str(candidate.reasoning),
            ]
        )
    return rows


def run_ranker(dataset_root: str, candidates_path: str) -> tuple[str, list[list[str]]]:
    root_text = dataset_root.strip()
    candidates_text = candidates_path.strip()

    try:
        root = _resolve_path(root_text, PROJECT_ROOT) or discover_dataset_root(PROJECT_ROOT)
        candidate_file = _resolve_path(candidates_text, root)
    except FileNotFoundError as exc:
        return f"{exc}", []

    if candidate_file is None:
        found = list(root.rglob("candidates.jsonl"))
        if not found:
            return "No candidates.jsonl file was found under the selected dataset root.", []
        candidate_file = found[0]

    job_text = load_job_description(root)
    jd_features = parse_job_description_features(job_text)
    embedder = SemanticEmbedder()

    pref_candidates = []
    for candidate in load_jsonl_candidates(candidate_file):
        candidate["_prefilter_score"] = cheap_prefilter_score(candidate, jd_features)
        pref_candidates.append(candidate)

    pref_candidates.sort(key=lambda item: (-float(item.get("_prefilter_score", 0.0)), str(item.get("candidate_id", ""))))
    shortlisted = pref_candidates[: min(len(pref_candidates), 100)]

    scored_candidates = []
    batch_items = []
    batch_texts = []
    batch_size = 128 if embedder.available else 256

    for candidate in shortlisted:
        profile_text = build_candidate_text(candidate)
        candidate["_profile_text"] = profile_text
        batch_items.append(candidate)
        batch_texts.append(profile_text)
        if len(batch_items) >= batch_size:
            similarities = embedder.batch_similarities(jd_features.raw_text, batch_texts)
            for item, similarity in zip(batch_items, similarities):
                scored_candidates.append(score_candidate(item, jd_features, similarity))
            batch_items = []
            batch_texts = []

    if batch_items:
        similarities = embedder.batch_similarities(jd_features.raw_text, batch_texts)
        for item, similarity in zip(batch_items, similarities):
            scored_candidates.append(score_candidate(item, jd_features, similarity))

    scored_candidates = normalize_and_sort(scored_candidates)

    if scored_candidates:
        top_candidate = scored_candidates[0]
        summary = (
            f"Ranked {len(scored_candidates):,} candidates. "
            f"Top candidate: {top_candidate.candidate_name or top_candidate.candidate_id} "
            f"({top_candidate.recommendation_tier}, {top_candidate.final_score:.2f})."
        )
    else:
        summary = "No candidates were ranked."

    return summary, _format_preview_rows(scored_candidates)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Candidate Ranking", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# AI Candidate Ranking\n"
            "Run the existing ranking pipeline from a browser and inspect the top candidates immediately."
        )

        with gr.Row():
            dataset_root = gr.Textbox(
                label="Dataset root",
                value="",
                placeholder="India_runs_data_and_ai_challenge",
            )
            candidates_path = gr.Textbox(
                label="Candidates JSONL path (optional)",
                placeholder="candidates.jsonl",
            )

        run_button = gr.Button("Run ranking", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Dataframe(
            headers=["Rank", "Candidate ID", "Candidate Name", "Current Title", "Score", "Tier", "Reasoning"],
            label="Top 10 preview",
            interactive=False,
            wrap=True,
        )

        run_button.click(fn=run_ranker, inputs=[dataset_root, candidates_path], outputs=[status, preview])

        gr.Markdown(
            "This UI only displays the ranking in the browser and does not write output files."
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()