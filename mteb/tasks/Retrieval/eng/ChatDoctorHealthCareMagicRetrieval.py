from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ChatDoctorHealthCareMagicRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChatDoctorHealthCareMagicRetrieval",
        dataset={
            "path": "embedding-benchmark/ChatDoctor_HealthCareMagic",
            "revision": "main",
        },
        description="ChatDoctor HealthCareMagic dataset for medical question answering retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/ChatDoctor_HealthCareMagic",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Non-fiction"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt={
            "query": "Given a medical question, retrieve relevant medical information and advice"
        },
    )