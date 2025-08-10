from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FinQARetrieval(AbsTaskRetrieval):
    metadata = {
        "name": "FinQARetrieval",
        "description": "FinQA is a dataset for financial question answering with numerical reasoning over financial documents.",
        "reference": "https://arxiv.org/abs/2109.00122",
        "dataset": {
            "path": "zeroshot/finqa-embedding-benchmark",
            "revision": "6b8d9d3df88b8b3eb7ec64e5e87e5b3076c16e3a",
            "trust_remote_code": True,
        },
        "type": "Retrieval",
        "category": "s2p",
        "modalities": ["text"],
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "ndcg_at_10",
        "revision": "6b8d9d3df88b8b3eb7ec64e5e87e5b3076c16e3a",
        "domains": ["Finance"],
        "task_subtypes": ["Question answering"],
        "license": "mit",
        "annotations_creators": "derived",
        "dialect": [],
        "sample_creation": "found",
        "bibtex_citation": """@article{chen2021finqa,
  title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
  author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  journal={arXiv preprint arXiv:2109.00122},
  year={2021}
}""",
        "descriptive_stats": {
            "n_samples": {"test": 1138},
            "avg_character_length": {
                "test": {
                    "average_document_length": 5679.147631043956,
                    "average_query_length": 125.55009649122808,
                    "num_documents": 380,
                    "num_queries": 1138,
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
