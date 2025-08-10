from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HC3FinanceRetrieval(AbsTaskRetrieval):
    metadata = {
        "name": "HC3FinanceRetrieval",
        "description": "HC3 Finance dataset for retrieval evaluation in financial domain.",
        "reference": "https://arxiv.org/abs/2301.07597",
        "dataset": {
            "path": "zeroshot/hc3finance-embedding-benchmark",
            "revision": "5e5d4171b86e0bb96b57159d991cbbbd73efcac0",
            "trust_remote_code": True,
        },
        "type": "Retrieval",
        "category": "s2p",
        "modalities": ["text"],
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "ndcg_at_10",
        "revision": "5e5d4171b86e0bb96b57159d991cbbbd73efcac0",
        "domains": ["Finance"],
        "task_subtypes": ["Question answering"],
        "license": "cc-by-sa-4.0",
        "annotations_creators": "derived",
        "dialect": [],
        "sample_creation": "found",
        "bibtex_citation": """@article{guo2023close,
  title={How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
  author={Guo, Biyang and Zhang, Xin and Yang, Zhiqi and Shi, Minqi and Wang, Tianxing and Yu, Dayiheng and Yao, Yuancheng},
  journal={arXiv preprint arXiv:2301.07597},
  year={2023}
}""",
        "descriptive_stats": {
            "n_samples": {"test": 415},
            "avg_character_length": {
                "test": {
                    "average_document_length": 748.9036144578313,
                    "average_query_length": 88.04819277108433,
                    "num_documents": 415,
                    "num_queries": 415,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    }

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
