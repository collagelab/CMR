import os
import pandas as pd
import json

if not os.path.exists("model_indices"):
    os.makedirs("model_indices")
    
    
required_files = [
    "./processed/cleaned-apibench-hf-train.json",
    "./processed/cleaned-mllm-train.json",
    "./processed/cleaned-hugging-bench-1-train.json",
    "./processed/cleaned-hugging-bench-2-train.json"
]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Required file {file} not found. Please run data processing script first.")

model_index = []    
for i, file in enumerate(required_files):
    df = pd.read_json(file, lines=True)
    
    try:
        model_index += list(set(df["api_data"].apply(lambda x: json.dumps(x)).unique().tolist()))
    except Exception as e:
        raise ValueError(f"Error processing file {file}: {e}")
    
    model_index_name = "_".join([f"e{j+1}" for j in range(i+1)])
    with open(f"./model_indices/{model_index_name}.json", "w") as f:
        for entry in model_index:
            f.write(entry+"\n")  
    print(f"Model index generated: ./model_indices/{model_index_name}.json")
    
        
        