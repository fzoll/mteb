from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CureV1EnRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CureV1EnRetrieval",
        dataset={
            "path": "embedding-benchmark/CureV1_en",
            "revision": "main",
        },
        description="CureV1 English dataset for medical question answering retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/CureV1_en",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Academic"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt={
            "query": "Given a medical question, retrieve relevant passages that best answer the question"
        },
    )