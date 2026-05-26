import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import tqdm

class SpladeRetriever:
    def __init__(self, corpus, query_model_id="naver/efficient-splade-VI-BT-large-query", doc_model_id="naver/efficient-splade-VI-BT-large-doc"):
        print("Building SPLADE index...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_id, cache_dir=os.environ["HF_HOME"])
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id, cache_dir=os.environ["HF_HOME"])
        self.query_model = AutoModelForMaskedLM.from_pretrained(query_model_id, cache_dir=os.environ["HF_HOME"]).to(self.device).eval()
        self.doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id, cache_dir=os.environ["HF_HOME"]).to(self.device).eval()
        self.corpus = corpus
        self.corpus_embeddings = self.encode_splade_batch(corpus, batch_size=32)

    def max_pool(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform SPLADEv2 max pooling operation.
        Returns a vocabulary-sized tensor with max-pooled values.
        """
        return torch.log1p(torch.relu(logits)).amax(dim=0)


    def encode_splade(self, text: str) -> torch.Tensor:
        """Encodes text into a sparse vector representation using SPLADE."""
        tokens = self.query_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,      # ensure we don’t exceed model max length
            max_length=512,       # DistilBERT/SPLADE supports 512 AT MOST
            padding="max_length"  # pad to same length
        ).to(self.device)
        with torch.no_grad():
            logits = self.query_model(**tokens).logits[0]  # tensor of shape [N, 30522]

        return self.max_pool(logits)  # tensor of size 30522

    def encode_splade_batch(self, texts, batch_size=64)-> torch.Tensor:
        """Encodes a batch of texts into sparse vectors using SPLADE."""
        all_vecs = []

        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Encoding SPLADE batch"):
            batch = texts[i:i + batch_size]
            tokens = self.doc_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,      # ensure we don’t exceed model max length
                max_length=512,       # DistilBERT/SPLADE supports 512 AT MOST
                padding="max_length"  # pad to same length
            ).to(self.device)

            with torch.no_grad():
                logits = self.doc_model(**tokens).logits

            # Apply SPLADE max pooling for each sequence in the batch
            batch_vecs = []
            for j in range(logits.size(0)):
                batch_vecs.append(self.max_pool(logits[j]))
            all_vecs.append(torch.stack(batch_vecs).cpu())  # Move to CPU immediately

        return torch.cat(all_vecs, dim=0)  # shape [num_texts, vocab_size]

    # def compute_score(self, query_vec: torch.Tensor, doc_vec: torch.Tensor) -> float:
    #     """
    #     Computes the dot product between the query and document tensors.
    #     The result is a similarity score.
    #     """
    #     return torch.dot(query_vec, doc_vec).item()

    def retrieve_model_card(self, instruction):
        query_vec = self.encode_splade(instruction).cpu().unsqueeze(0)
        scores = torch.mv(self.corpus_embeddings, query_vec.squeeze(0))
        best_idx = torch.argmax(scores).item()
        return self.corpus[best_idx]