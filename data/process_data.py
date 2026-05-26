import argparse
import os
from .utils.hugging_bench import process_hugging_bench
from .utils.apibench_hf import process_apibench_hf
from .utils.mllm import process_mllm
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login

from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (one level up from data/process_data.py)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


#set huggingface cache, sentence transformers, tokenizer cache
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.hf_cache"))
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["TOKENIZERS_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir


def main():
    """Main function to handle command line arguments and process data."""
    parser = argparse.ArgumentParser(
        description="Process different datasets for the CMR benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Available data options:
                apibench-hf   Process HuggingFace API benchmark data
                mllm          Process MLLM text-to-text data
                hugging-bench-1    Process Hugging Bench part one data
                hugging-bench-2    Process Hugging Bench part two data

                Examples:
                python process_data.py --data apibench-hf
                python process_data.py --data mllm
                python process_data.py --data hugging-bench-1
                python process_data.py --data hugging-bench-2
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["apibench-hf", "mllm", "hugging-bench-1", "hugging-bench-2"],
        help="Specify which dataset to process"
    )
    
    args = parser.parse_args()
        
    token = os.getenv("HF_API_TOKEN")
    if token:
        login(token=token)
    else:
        raise ValueError("Hugging Face token is required for authentication. Set HF_API_TOKEN in .env file.")
   
    raw_data_dir = Path(__file__).resolve().parents[1] / "data/raw"
    processed_data_dir = Path(__file__).resolve().parents[1] / "data/processed"

    # Ensure processed data directory exists
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    columns_to_keep = ["model_name", "model_family", "created_at", "instruction", "domain", "original_dataset", "model_source", "api_data", "explanation"]
     
    # Process data based on the argument
    if args.data == "apibench-hf":
        dfs = process_apibench_hf(raw_data_dir)
    
    elif args.data == "mllm":
        dfs = process_mllm(raw_data_dir)
    
    elif args.data == "hugging-bench-1" or args.data == "hugging-bench-2":
        if args.data == "hugging-bench-1":
            dfs = process_hugging_bench(raw_data_dir, part=1)
        else:
            dfs = process_hugging_bench(raw_data_dir, part=2)
    
    else:
        raise ValueError(f"Unsupported data option: {args.data}")
    
    # null value check
    for df in dfs:
        if df.isnull().values.any():
            print(df.info())
            raise ValueError("DataFrame contains null values")
    
    for i, df in enumerate(dfs):
        dfs[i] = df[columns_to_keep]
        
    train_df, val_df = train_test_split(
        dfs[0], 
        test_size=0.15, 
        random_state=42, 
        stratify=dfs[0]['model_name']
    )
    
    # Save train, validation, and eval datasets
    train_df.to_json(f"{processed_data_dir}/cleaned-{args.data}-train.json", orient="records", lines=True)
    val_df.to_json(f"{processed_data_dir}/cleaned-{args.data}-val.json", orient="records", lines=True)
    
    #dfs[0].to_json(f"{processed_data_dir}/cleaned-{args.data}-train.json", orient="records", lines=True)
    dfs[1].to_json(f"{processed_data_dir}/cleaned-{args.data}-eval.json", orient="records", lines=True)

    print(f"Data processing completed for: {args.data}")


if __name__ == "__main__":
    main()
