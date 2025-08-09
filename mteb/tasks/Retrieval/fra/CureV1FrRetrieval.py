from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CureV1FrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CureV1FrRetrieval",
        dataset={
            "path": "embedding-benchmark/CureV1_fr",
            "revision": "main",
        },
        description="CureV1 French dataset for medical question answering retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/CureV1_fr",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["fra-Latn"],
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
            "query": "Étant donné une question médicale, récupérez les passages pertinents qui répondent le mieux à la question"
        },
    )