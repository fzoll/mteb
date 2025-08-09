from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DS1000Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DS1000Retrieval",
        dataset={
            "path": "embedding-benchmark/DS1000",
            "revision": "main",
        },
        description="DS1000 dataset for data science code retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/DS1000",
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
            "query": "Given a data science problem, retrieve relevant code examples or library documentation"
        },
    )