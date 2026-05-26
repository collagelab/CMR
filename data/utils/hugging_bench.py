from .utility import (
    load_dataframe, 
    get_description_from_model_card, 
    categories, 
    get_closest_functionality, 
    SLEEP_TIME, 
    inject_model_name,
    normalize_model_name,
    add_model_family,
)
from huggingface_hub import HfApi, ModelCard
import time
import pandas as pd

def get_api_data_for_hugging_bench(hugging_bench_df, apibench_df, mllm_df):
    functionalities_set = set()
    for category, values in categories.items():
        for v in values:
            functionalities_set.add(v)
    
    api = HfApi()
    apibench_models = apibench_df["model_name"].unique()
    mllm_models = mllm_df["model_name"].unique()
    
    models = hugging_bench_df["model_name"].unique()
    
    model_to_api_data = {}

    for model in models:
        if model in apibench_models:
            model_to_api_data[model] = apibench_df[apibench_df["model_name"]==model].iloc[0]["api_data"]
            print(f"Found API data for model '{model}' in APIBench.")
            continue
        
        elif model in mllm_models:
            model_to_api_data[model] = mllm_df[mllm_df["model_name"]==model].iloc[0]["api_data"]
            print(f"Found API data for model '{model}' in MLLM.") 
            continue
        
        else:
            print(f"Fetching API data for model '{model}' from Hugging Face...")
            d = {}
        
        try:
            model_info = api.model_info(model)

            if model_info.pipeline_tag:
                functionality = model_info.pipeline_tag.replace("-", " ")

            elif model_info.tags:
                functionality = get_closest_functionality(model_info.tags, functionalities_set)

            else:
                raise ValueError("Cannot get functionality: pipeline_tag is None.")


            # search in categories values list the functionality and take the key name
            for category, values in categories.items():
                for v in values:
                    if functionality.lower() == v.lower().replace("-", " "):
                        #d["functionality"] = v
                        d["domain"] = f"{category} {v}"
                        break
            
            # model_info may have 'library_name' can be None.
            # Safely handle missing/None values by appending an empty string instead of raising.
            lib_name = None
            if hasattr(model_info, "library_name"):
                lib_name = model_info.library_name
            # Normalize to string and title-case when present, otherwise use empty string.
            lib_str = "" if lib_name is None else str(lib_name).title()
            d["framework"] = f"Hugging Face {lib_str}".strip()
            
            d["api_name"] = model
            d["performance"] = {}
            model_card = ModelCard.load(model, ignore_metadata_errors=True)
            # Safely extract `dataset` from model_card.data. If missing or any
            # error occurs, default to empty string instead of raising.
            try:
                data = getattr(model_card, "data", None)
                if data is None:
                    dataset_value = ""
                else:
                    try:
                        data_dict = data.to_dict()
                        dataset_value = data_dict.get("dataset", "") if isinstance(data_dict, dict) else ""
                    except Exception:
                        # Fallback: try attribute access or mapping conversion
                        dataset_value = getattr(data, "dataset", "") or ""
                d["performance"]["dataset"] = dataset_value
            except Exception:
                d["performance"]["dataset"] = ""
            d["performance"]["accuracy"] = None
            d["description"] = get_description_from_model_card(model_card)

            d["api_call"] = ""

            model_to_api_data[model] = d
        except Exception as e:
            print(f"Failed to fetch info for model '{model}'. {e}\n")
            model_to_api_data[model] = pd.NA
        time.sleep(SLEEP_TIME)  # To avoid hitting API rate limits
    
    return model_to_api_data


def process_hugging_bench(raw_data_dir, part=1):
    """Process Hugging Bench data."""
    print("Processing Hugging Bench data...")

    processed_data_dir = "./data/processed"
    if part ==  1:
        index = 3
    else:
        index = 4

    train_df = load_dataframe(f"{raw_data_dir}/exp{index}/exp{index}-train.jsonl", lines=True)
    val_df = load_dataframe(f"{raw_data_dir}/exp{index}/exp{index}-val.jsonl", lines=True)
    
    # concatenate train and val
    train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    eval_df = load_dataframe(f"{raw_data_dir}/exp{index}/exp{index}-eval.jsonl", lines=True)
    processed_dfs = []

    for i, df in enumerate([train_df, eval_df]):
        if i == 0: # train
            apibench_df = load_dataframe(f"{processed_data_dir}/cleaned-apibench-hf-train.json", lines=True)
            mllm_df = load_dataframe(f"{processed_data_dir}/cleaned-mllm-train.json", lines=True)
            
        else: # eval
            apibench_df = load_dataframe(f"{processed_data_dir}/cleaned-apibench-hf-eval.json", lines=True)
            mllm_df = load_dataframe(f"{processed_data_dir}/cleaned-mllm-eval.json", lines=True)
            
        df['original_dataset'] = "Hugging Bench"
        df['model_source'] = "HuggingFace"
        df['explanation'] = ""
        
        # prepend "###Instruction:" in instruction column
        df['instruction'] = df['instruction'].apply(lambda x: f"###Instruction: {x}")
        
        # Add normalized_model_name for family computation
        df['normalized_model_name'] = df['model_name'].apply(normalize_model_name)
        
        model_to_api_data = get_api_data_for_hugging_bench(df, apibench_df, mllm_df)
        df["api_data"] = df["model_name"].map(model_to_api_data)
        df = df.dropna().copy() # we can have NA in api_data since some models may not be found anymore on HF
        
        df['api_data'] = df.apply(
            lambda row: inject_model_name(row['api_data'], row['model_name']),
            axis=1,
        )
        
        df['domain'] = df['api_data'].apply(lambda x: x.get('domain'))
        
        processed_dfs.append(df)
    
    if part == 1:
        previous_train_df = load_dataframe(f"{processed_data_dir}/cleaned-mllm-train.json", lines=True)
    else:
        previous_train_df = load_dataframe(f"{processed_data_dir}/cleaned-hugging-bench-1-train.json", lines=True)
        
    processed_dfs = add_model_family(processed_dfs, previous_train_df=previous_train_df)
    
    print(f"Hugging Bench part {part} data processed successfully.")
    return processed_dfs