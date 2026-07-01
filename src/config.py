from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_DATASET_GLOB = "**/candidates.jsonl"
DEFAULT_JOB_GLOB = "**/job_description.docx"
DEFAULT_SAMPLE_CANDIDATES_GLOB = "**/sample_candidates.json"
DEFAULT_SAMPLE_SUBMISSION_GLOB = "**/sample_submission.csv"
DEFAULT_SCHEMA_GLOB = "**/candidate_schema.json"
DEFAULT_REPORT_NAME = "report.html"
DEFAULT_OUTPUT_NAME = "ranked_candidates.csv"
DEFAULT_TOP_N = 100
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AI_KEYWORDS = {
    "ai",
    "ml",
    "machine learning",
    "llm",
    "large language model",
    "embeddings",
    "retrieval",
    "ranking",
    "search",
    "recommendation",
    "fine-tuning",
    "fine tuning",
    "rag",
    "vector database",
    "vector search",
    "prompt engineering",
    "evaluation",
    "inference",
    "nlp",
    "transformers",
    "pytorch",
    "tensorflow",
    "python",
    "sql",
    "spark",
    "airflow",
    "mlops",
    "bentoml",
}

SKILL_SYNONYMS = {
    "python": ["python", "py", "python3"],
    "sql": ["sql", "postgres", "mysql", "sqlite", "warehouse", "snowflake", "bigquery"],
    "spark": ["spark", "pyspark", "spark streaming"],
    "airflow": ["airflow", "dag", "workflow orchestration"],
    "llm": ["llm", "large language model", "llms", "gpt", "fine-tuning llms", "fine tuning llms"],
    "embeddings": ["embedding", "embeddings", "sentence transformers", "vector embedding"],
    "retrieval": ["retrieval", "search", "vector search", "semantic search", "ranking"],
    "ranking": ["ranking", "ranker", "re-ranking", "reranking", "re rank", "scorecard"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "fine_tuning": ["fine-tuning", "fine tuning", "lora", "qlora", "sft", "instruction tuning"],
    "nlp": ["nlp", "natural language processing", "text mining"],
    "mlops": ["mlops", "bentoml", "mlflow", "wandb", "model deployment"],
    "cloud": ["aws", "gcp", "azure", "cloud", "docker", "kubernetes"],
    "data_engineering": ["dbt", "etl", "elt", "pipelines", "feature pipelines", "data pipelines"],
    "backend": ["backend", "api", "flask", "fastapi", "service"],
    "evaluation": ["evaluation", "metrics", "benchmark", "ab test", "experimentation"],
    "computer_vision": ["image classification", "cv", "computer vision", "vision"],
    "speech": ["speech recognition", "tts", "asr", "speech", "audio"],
    "vector_db": ["milvus", "faiss", "pinecone", "weaviate", "vector database", "vector db"],
}

FAMILY_MAP = {
    "python": "engineering",
    "sql": "engineering",
    "spark": "engineering",
    "airflow": "engineering",
    "llm": "ai",
    "embeddings": "ai",
    "retrieval": "ai",
    "ranking": "ai",
    "rag": "ai",
    "fine_tuning": "ai",
    "nlp": "ai",
    "mlops": "platform",
    "cloud": "platform",
    "data_engineering": "platform",
    "backend": "platform",
    "evaluation": "product",
    "computer_vision": "ai",
    "speech": "ai",
    "vector_db": "ai",
}

TITLE_SENIORITY = {
    "intern": 0,
    "junior": 1,
    "associate": 1,
    "trainee": 0,
    "engineer": 2,
    "developer": 2,
    "analyst": 2,
    "scientist": 3,
    "senior": 3,
    "lead": 4,
    "staff": 4,
    "principal": 4,
    "manager": 3,
    "director": 5,
    "vp": 6,
    "head": 5,
    "founder": 6,
}

DEGREE_SCORES = {
    "phd": 5.0,
    "doctor": 5.0,
    "m.tech": 4.2,
    "mtech": 4.2,
    "m.sc": 3.8,
    "msc": 3.8,
    "m.e.": 3.8,
    "me": 3.8,
    "mba": 2.5,
    "b.tech": 3.0,
    "btech": 3.0,
    "b.e.": 3.0,
    "be": 3.0,
    "bsc": 2.2,
    "other": 1.0,
}

FIELD_HINTS = {
    "computer science",
    "information technology",
    "artificial intelligence",
    "machine learning",
    "data science",
    "software engineering",
    "electronics",
    "statistics",
    "mathematics",
    "physics",
    "engineering",
}
