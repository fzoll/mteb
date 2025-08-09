from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MBPPRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MBPPRetrieval",
        dataset={
            "path": "embedding-benchmark/MBPP",
            "revision": "main",
        },
        description="MBPP (Mostly Basic Python Problems) dataset for code retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/MBPP",
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
            "query": "Given a Python programming problem description, retrieve relevant code examples or solutions"
        },
    )