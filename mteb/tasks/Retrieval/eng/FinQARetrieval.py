from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FinQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FinQARetrieval",
        description="Large-scale dataset focused on financial reasoning, containing 2.8k financial reports and 8k question-and-answer pairs",
        reference="https://huggingface.co/datasets/embedding-benchmark/FinQA",
        dataset={
            "path": "embedding-benchmark/FinQA",
            "revision": "bdd1903",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Finance"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        from datasets import load_dataset

        # Load the three configurations
        corpus_ds = load_dataset(self.metadata_dict["dataset"]["path"], "corpus")[
            "corpus"
        ]
        queries_ds = load_dataset(self.metadata_dict["dataset"]["path"], "queries")[
            "queries"
        ]
        qrels_ds = load_dataset(self.metadata_dict["dataset"]["path"], "default")[
            "test"
        ]

        # Process corpus
        for item in corpus_ds:
            self.corpus[item["id"]] = {"title": "", "text": item["text"]}

        # Process queries
        for item in queries_ds:
            self.queries[item["id"]] = item["text"]

        # Process qrels (relevant documents)
        for item in qrels_ds:
            query_id = item["query_id"]
            if query_id not in self.relevant_docs:
                self.relevant_docs[query_id] = {}
            self.relevant_docs[query_id][item["corpus_id"]] = item["score"]

        self.data_loaded = True
