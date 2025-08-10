from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DS1000Retrieval(AbsTaskRetrieval):
    metadata = {
        "name": "DS1000Retrieval",
        "description": "DS-1000 is a code generation dataset focused on data science tasks, adapted for retrieval evaluation.",
        "reference": "https://arxiv.org/abs/2211.11501",
        "dataset": {
            "path": "zeroshot/ds1000-embedding-benchmark",
            "revision": "5e5d4171b86e0bb96b57159d991cbbbd73efcac0",
            "trust_remote_code": True,
        },
        "type": "Retrieval",
        "category": "s2s",
        "modalities": ["text"],
        "eval_splits": ["test"],
        "eval_langs": ["eng-Latn"],
        "main_score": "ndcg_at_10",
        "revision": "5e5d4171b86e0bb96b57159d991cbbbd73efcac0",
        "domains": ["Programming"],
        "task_subtypes": ["Code retrieval"],
        "license": "mit",
        "annotations_creators": "derived",
        "dialect": [],
        "sample_creation": "found",
        "bibtex_citation": """@article{lai2022ds,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Lai, Yuhang and Li, Chengxi and Wang, Yiming and Zhang, Tianyi and Zhong, Ruiqi and Zettlemoyer, Luke and Yih, Scott Wen-tau and Fried, Daniel and Wang, Sida I and Yu, Tao},
  journal={arXiv preprint arXiv:2211.11501},
  year={2022}
}""",
        "descriptive_stats": {
            "n_samples": {"test": 1998},
            "avg_character_length": {
                "test": {
                    "average_document_length": 260.2947947947948,
                    "average_query_length": 419.0765765765766,
                    "num_documents": 1998,
                    "num_queries": 1998,
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
