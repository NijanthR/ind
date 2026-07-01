# AI Candidate Ranking System

End-to-end candidate ranking pipeline for the Redrob hiring challenge.

## What it does

- Reads the extracted dataset bundle.
- Parses the job description and candidate JSONL at scale.
- Computes semantic similarity with `sentence-transformers` when available, with a deterministic hashed fallback.
- Scores structured signals for skills, experience, education, platform activity, and career trajectory.
- Writes a validator-compliant CSV and a self-contained HTML report.

## Files

- Source code: `src/`
- Submission CSV: `output/ranked_candidates.csv`
- Report: `output/report.html`

## Setup

1. Create a virtual environment if needed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` if you want to record local runtime settings.

## Run

From the workspace root:

```bash
.venv\Scripts\python.exe src\main.py
```

To launch the Gradio UI instead:

```bash
.venv\Scripts\python.exe src\gradio_app.py
```

If you omit the dataset path in the UI, it auto-discovers the dataset bundle under the workspace.
The Gradio UI only displays the ranking in the browser and does not write CSV or HTML output files.

## Validation

The repository includes `validate_submission.py` from the challenge bundle. The generated CSV follows that validator's required 4-column schema and 100-row ranking contract.

## Notes

- The detailed scoring breakdown appears in `output/report.html`.
- The pipeline is fully offline and deterministic.
