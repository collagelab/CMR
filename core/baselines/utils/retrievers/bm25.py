from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus: list[str]):
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.corpus = corpus

    def retrieve_model_card(self, instruction: str) -> str:
        tokenized_query = instruction.split(" ")
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=1)
        return bm25_docs[0]