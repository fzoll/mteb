from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WikiSQLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WikiSQLRetrieval",
        dataset={
            "path": "embedding-benchmark/WikiSQL",
            "revision": "main",
        },
        description="WikiSQL dataset for SQL query retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/WikiSQL",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt={
            "query": "Given a natural language question, retrieve relevant SQL queries and database schemas"
        },
    )