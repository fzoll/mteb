from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HumanEvalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HumanEvalRetrieval",
        dataset={
            "path": "embedding-benchmark/HumanEval",
            "revision": "main",
        },
        description="HumanEval dataset for code understanding and retrieval tasks.",
        reference="https://huggingface.co/datasets/embedding-benchmark/HumanEval",
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
            "query": "Given a programming question or code description, retrieve relevant code examples or documentation"
        },
    )