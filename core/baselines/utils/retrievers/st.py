from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import os

class SentenceTransformerRetriever:
  def __init__(self, corpus, model_name="all-mpnet-base-v2"):
    print("Building SentenceTransformer index...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = SentenceTransformer(model_name, device=device, cache_folder=os.environ["HF_HOME"])
    self.device = device
    list_of_text = [text.replace("\n", " ") for text in corpus]
    self.corpus_embeddings = self.model.encode(list_of_text, convert_to_tensor=True, device=self.device)
    self.corpus = corpus

  def retrieve_model_card(self, instruction):
    query_embedding = self.model.encode(instruction, convert_to_tensor=True, device=self.device)

    similarity = util.cos_sim(query_embedding, self.corpus_embeddings)
    max_index = similarity.argmax()

    return self.corpus[max_index]