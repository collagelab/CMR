import json
from typing import Dict, List

from datasets import Dataset
from tqdm import tqdm
import gc
import torch

from .configs import ModelIndicesDataConfig
from .retrievers.bgem3_reranker import HybridBGEM3Retriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.splade import SpladeRetriever
from .retrievers.st import SentenceTransformerRetriever

gorilla_prompt = (
    "You are Gorilla, an expert API model router. "
    "Read the ###Instruction and ###Input below and return ONLY a single model name. "
    "Do not invent model name. Do not return anything else.\n\n"
)

gorilla_prompt_with_retrieval = gorilla_prompt + ("In this task, you have access to suggested API information retrieved from a knowledge base. Not always the retrieved information is relevant or accurate. Use your judgment to decide whether to incorporate it into your response. You will find retrieved API information appended after the instruction under <Reference API> tag.\n\n")


def load_dataset_json(path: str) -> list:
    dataset_json = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_json.append(json.loads(line))
            else:
                raise ValueError("Empty line found in the dataset file.")
    return dataset_json


def get_prompt(instruction: str, system_prompt: str = gorilla_prompt, retriever: BM25Retriever | SentenceTransformerRetriever | SpladeRetriever | HybridBGEM3Retriever | None = None) -> str:
    model_card = ""

    # Retrieve model_card
    if retriever is not None:
        try:
            retrieved_info = retriever.retrieve_model_card(instruction)
            model_card = (
                "\n<Reference API>: " + retrieved_info.replace("\r\n", "\n").strip()
            )
        except Exception as e:
            raise ValueError(f"Error retrieving model card: {e}")

    if instruction:
        # Build the full prompt without extra stripping
        full_prompt = system_prompt + instruction + model_card + "\n###Response:"
        return full_prompt
    else:
        raise ValueError(
            "Both 'instruction' and 'model_name' must be present in each data entry."
        )

def get_retriever(retriever_name: str, model_index_name: str) -> BM25Retriever | SentenceTransformerRetriever | SpladeRetriever | HybridBGEM3Retriever | None:
    retriever = None
     
    model_index = load_dataset_json(ModelIndicesDataConfig().get_model_index_path(model_index_name))
    # model_index is a list of dict, convert into a list of strings
    model_index = [json.dumps(entry) for entry in model_index]
    print(
        f"Loading model index {ModelIndicesDataConfig().get_model_index_path(model_index_name)} model index with {len(model_index)} entries for retrieval."
    )

    if retriever_name == "bm25":
        retriever = BM25Retriever(model_index)
    elif retriever_name == "sentence_transformer":
        retriever = SentenceTransformerRetriever(model_index)
    elif retriever_name == "splade":
        retriever = SpladeRetriever(model_index)
    elif retriever_name == "flagembedding":
        retriever = HybridBGEM3Retriever(model_index)
    else:
        raise ValueError(
            f"Retriever '{retriever_name}' is not supported. Choose from: ['bm25', 'sentence_transformer', 'splade', 'flagembedding']"
        )
            
    return retriever

def convert_to_conversational(
    raw_data: List[Dict[str, str]],
    tokenizer,
    model_index_name: str | None,
    retriever_name: str | None,
    old_model_ids: list = None,
) -> List[Dict[str, str]]:
    conversational_dataset = []
    system_prompt = gorilla_prompt
    
    retriever = None
    if model_index_name is not None and retriever_name is not None:
        system_prompt = gorilla_prompt_with_retrieval
        retriever = get_retriever(retriever_name, model_index_name)
    

    for entry in tqdm(raw_data, desc="Converting to conversational format"):
        prompt = entry.get("instruction", "").replace("\r\n", "\n").strip()
        answer = entry.get("model_name", "").replace("\r\n", "\n").strip()

        full_prompt = get_prompt(prompt, system_prompt=system_prompt, retriever=retriever) # retriever can be None
        conversational_dataset.append(
            {
                "prompt": full_prompt,
                "completion": " " + answer + tokenizer.eos_token,  # Keep consistent leading space
                "is_old": (answer in old_model_ids) if old_model_ids is not None else False,
            }
        )
    
    if retriever is not None:
        del retriever  # free up memory
        gc.collect()
        torch.cuda.empty_cache()
        
    print("Example converted data point:")
    print(conversational_dataset[0] if conversational_dataset else "No data points converted.")

    dataset = Dataset.from_list(conversational_dataset)
    return dataset
