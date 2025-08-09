from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HC3FinanceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HC3FinanceRetrieval",
        dataset={
            "path": "embedding-benchmark/HC3Finance",
            "revision": "main",
        },
        description="HC3 Finance dataset for retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/HC3Finance",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt={
            "query": "Given a finance-related question, retrieve relevant documents that best answer the question"
        },
    )