from .utility import (
    load_dataframe, 
    extract_instruction_apibench, 
    extract_model_name_from_api_call, 
    remove_fields_from_api_data, 
    get_model_dates, 
    inject_model_name,
    normalize_model_name,
    add_model_family,
)

import pandas as pd
import re
#from huggingface_hub import HfApi

        
def process_apibench_hf(raw_data_dir):
    """Process HuggingFace API benchmark data."""
    print("Processing HuggingFace API benchmark data...")

    train_df = load_dataframe(f"{raw_data_dir}/apibench/huggingface_train.json", lines=True)
    eval_df = load_dataframe(f"{raw_data_dir}/apibench/huggingface_eval.json", lines=True)
    processed_dfs = []
       
    # api = HfApi()

    for df in [train_df, eval_df]:
        # We can't extact model_name from api_data because most of the time api_name field does not contain repo owner id
        # repo_id is needed because we need to check which of APIBench models are present in MLLM dataset
        # So we extract model_name from api_call instead
                  
        df['model_name'] = df["api_call"].apply(lambda api_call: extract_model_name_from_api_call(api_call))
        df = df.dropna(subset=['model_name']).copy()

        models = df["model_name"].unique()
        model_to_date = get_model_dates(models, cutoff_date=pd.Timestamp("2023-05-24", tz="UTC"))

        # Map model_name to created_at (may be None)
        df["created_at"] = df["model_name"].map(model_to_date)

        # Normalize to datetimes (coerce invalid/missing values to NaT)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df["created_at"] = df["created_at"].fillna(df["created_at"].mean())
        
        # df = df.dropna(subset=['created_at']).copy()
        # df["created_at"] = pd.Timestamp("2023-05-25", tz="UTC")
        
        df["api_name"] = df["api_data"].apply(lambda x: x.get("api_name"))

        df['instruction'] = df['code'].apply(extract_instruction_apibench)
        df['domain'] = df['api_data'].apply(lambda x: x.get('domain'))

        df['original_dataset'] = "APIBench"
        df['model_source'] = "HuggingFace"

        df['api_data'] = df['api_data'].apply(remove_fields_from_api_data)


        df['api_data'] = df.apply(
            lambda row: inject_model_name(row['api_data'], row['model_name']),
            axis=1,
        )
        

        df['explanation'] = df["code"].str.extract(
            r"<<<explanation>>>:(.*?)(?=<<<code>>>|$)",  # stops at <<<code>>> or end of string
            flags=re.S
            ).fillna("")
        
        # this is needed to compute model family later
        df["normalized_model_name"] = df["model_name"].apply(normalize_model_name)
        
        processed_dfs.append(df)
    
    processed_dfs = add_model_family(processed_dfs)
    
    print("Finished processing HuggingFace API benchmark data")
    return processed_dfs