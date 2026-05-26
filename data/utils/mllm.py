import pandas as pd
from .utility import (
    load_dataframe,
    get_description_from_model_card,
    categories,
    get_closest_functionality,
    get_model_dates,
    SLEEP_TIME,
    inject_model_name,
    normalize_model_name,
    add_model_family,
)
from huggingface_hub import HfApi, ModelCard
import os
import time
from sklearn.model_selection import train_test_split

  

def get_api_data_for_mllm(mllm_df, apibench_df):
    functionalities_set = set()
    for category, values in categories.items():
        for v in values:
            functionalities_set.add(v)
    
    api = HfApi()
    apibench_models = apibench_df["model_name"].unique()
    apibench_domains = apibench_df["domain"].unique()
    
    models = mllm_df["model_name"].unique()
    model_to_api_data = {}

    for model in models:
        if model in apibench_models:
            model_to_api_data[model] = apibench_df[apibench_df["model_name"]==model].iloc[0]["api_data"]
            print(f"Found API data for model '{model}' in APIBench.")
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
            
            if "domain" not in d:
                values = categories[functionality]
                # get closest functionalities with apibench domains without threshold
                closest_func = get_closest_functionality(values, apibench_domains, threshold=1)
                d["domain"] = closest_func
                
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


def process_mllm(raw_data_dir):
    """Process MLLM text-to-text data."""
    print("Processing MLLM text-to-text data...")


    # check if "cleaned-apibench-hf-train.json" exists in processed dir, if not raise error
    processed_data_dir = "./data/processed"
    if not os.path.exists(f"{processed_data_dir}/cleaned-apibench-hf-train.json"):
        raise FileNotFoundError(f"{processed_data_dir}/cleaned-apibench-hf-train.json not found. Please process APIBench data first.")

    apibench_df = load_dataframe(f"{processed_data_dir}/cleaned-apibench-hf-train.json", lines=True)
    
    df_text_t2t = load_dataframe(f"{raw_data_dir}/mllm/text_t2t.json", lines=False)
    df_image_t2t = load_dataframe(f"{raw_data_dir}/mllm/image_tx2t.json", lines=False)
    df_video_t2t = load_dataframe(f"{raw_data_dir}/mllm/video_tx2t.json", lines=False)
    df_audio_t2t = load_dataframe(f"{raw_data_dir}/mllm/audio_tx2t.json", lines=False)
    
    # restrict to only conversations column for both dataframes
    df_text_t2t = df_text_t2t[["conversations"]]
    df_image_t2t = df_image_t2t[["conversations"]]
    df_video_t2t = df_video_t2t[["conversations"]]
    df_audio_t2t = df_audio_t2t[["conversations"]]
    
    df = pd.concat([df_text_t2t, df_image_t2t, df_video_t2t, df_audio_t2t], ignore_index=True)
    
    df["model_name"] = df["conversations"].apply(
        lambda x: next((d["value"].strip() for d in x if d.get("from") == "gpt"), None)
    )
     
    # filter out "unknown" models
    df = df[df["model_name"] != "unknown"].copy()
    
    # Add normalized_model_name for family computation
    df['normalized_model_name'] = df['model_name'].apply(normalize_model_name)
   
    df.loc[:, "instruction"] = df["conversations"].apply(
        lambda x: (lambda s: f"###Instruction: {s}" if s is not None else None)(
            next((d["value"] for d in x if d.get("from") == "human"), None)
        )
    )
    
    if df["instruction"].isnull().any():
        raise ValueError("Some instructions are None. Please check the data.")

    # count model occurrences
    df.loc[:, "model_occurs"] = df.groupby("model_name")["model_name"].transform("count")

    # sort, deduplicate, and recalc counts
    df = df.sort_values(by="model_occurs", ascending=True)
    df = df.drop_duplicates(subset=["instruction"], keep="first")
    df.loc[:, "model_occurs"] = df.groupby("model_name")["model_name"].transform("count")

    # filter rare models
    df = df[df["model_occurs"] >= 7].copy()

    df['original_dataset'] = "MLLM"
    df['model_source'] = "HuggingFace"
    
    models = df["model_name"].unique()
    model_to_date = get_model_dates(models, cutoff_date=pd.Timestamp("2024-01-19", tz="UTC"))

    # Map model_name to created_at (may be None)
    df["created_at"] = df["model_name"].map(model_to_date)

    # Normalize to datetimes (coerce invalid/missing values to NaT)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["created_at"] = df["created_at"].fillna(df["created_at"].mean())
    
    #df["created_at"] = pd.Timestamp("2024-01-19", tz="UTC") # this is just to test, needs to be changes with actual data
    
    df['explanation'] = ""

    model_to_api_data = get_api_data_for_mllm(df, apibench_df)
    df["api_data"] = df["model_name"].map(model_to_api_data)
    df = df.dropna().copy() # we can have NA in api_data since some models may not be found anymore on HF
    
    df['api_data'] = df.apply(
            lambda row: inject_model_name(row['api_data'], row['model_name']),
            axis=1,
        )

    df['domain'] = df['api_data'].apply(lambda x: x.get('domain'))
    
    # check if df has nones in all columns, if yes print rows with nones
    if df.isnull().all().any():
        print("Rows with all None values in at least one column:")
        print(df[df.isnull().all(axis=1)])
        raise ValueError("DataFrame has all None values in at least one column.")
    
     
    # Split maintaining the distribution of model_name
    train_df, eval_df = train_test_split(
        df, 
        test_size=0.15, 
        random_state=42, 
        stratify=df['model_name']
    )
    
    # Compute model families in continual learning setting
    # using previous experience (apibench-hf train) + current experience
    processed_dfs = add_model_family([train_df, eval_df], apibench_df)

    return processed_dfs