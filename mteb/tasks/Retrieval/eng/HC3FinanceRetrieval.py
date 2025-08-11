from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HC3FinanceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HC3FinanceRetrieval",
        description="Text retrieval collection focused on comparing human and AI-generated responses across various domains, particularly in finance",
        reference="https://huggingface.co/datasets/embedding-benchmark/HC3Finance",
        dataset={
            "path": "embedding-benchmark/HC3Finance",
            "revision": "fda6fad",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Finance"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        from datasets import load_dataset

        # Load the three configurations
        corpus_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "corpus",
            revision=self.metadata_dict["dataset"]["revision"],
        )["corpus"]
        queries_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "queries",
            revision=self.metadata_dict["dataset"]["revision"],
        )["queries"]
        qrels_ds = load_dataset(
            self.metadata_dict["dataset"]["path"],
            "default",
            revision=self.metadata_dict["dataset"]["revision"],
        )["test"]

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
