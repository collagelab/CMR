import json
import logging
import os
import warnings
from .utils.eval_utility import compute_metrics
from .utils.configs import (
    ApibenchDataConfig,
    MLLMDataConfig,
    HuggingBench1DataConfig,
    HuggingBench2DataConfig,
    EvalConfig,
)
from ..baselines.utils.prepareDataset import load_dataset_json, get_prompt, gorilla_prompt, gorilla_prompt_with_retrieval, get_retriever
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from .utils.parser import EvalParser
from .openmodel import LoRAModelManager


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")

cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.hf_cache"))
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["TOKENIZERS_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

# Suppress all unnecessary logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

MODEL_INDICES = []
RETRIEVERS = []
EXPERIENCES = []
LORA_ADAPTERS_MAP = {
    "apibench": "",
    "mllm": "",
    "hugging-bench-1": "",
    "hugging-bench-2": "",
}
LORA_WEIGHTS = []


def get_answers(
    question_jsons: list[dict],
    model: LoRAModelManager | None,
    retriever_name: str | None,
    model_index_name: str | None,
    eval_config: EvalConfig,
) -> list[dict]:
    
    retriever = get_retriever(retriever_name, model_index_name) if retriever_name is not None and model_index_name is not None else None
      
    ans_jsons = []
    if retriever_name is None:
        system_prompt = gorilla_prompt
    else:
        system_prompt = gorilla_prompt_with_retrieval
    prompts = []


    for q_json in tqdm(question_jsons, desc="Get prompts or get retrieved info"):
        instruction = q_json.get("instruction", "").strip().replace("\r\n", "\n")
        
        if model is not None:
            prompt = get_prompt(instruction, system_prompt=system_prompt, retriever=retriever) # retriever can be None
            
            # Ensure prompt is a string
            if not isinstance(prompt, str):
                raise TypeError(
                    f"get_prompt returned {type(prompt)}, expected str. Value: {prompt}"
                )

            prompts.append(prompt)
        else: # retrievers only mode
            if eval_config.oracle:
                retrieved_info = json.loads(retriever.retrieve_model_card_with_oracle(instruction, q_json.get("domain", "")))
            else:
                retrieved_info = json.loads(retriever.retrieve_model_card(instruction))
            ans_jsons.append(
                {
                    "questions": q_json["instruction"],
                    "response": retrieved_info["model_name"].strip(),
                    "ground_true": q_json["model_name"].strip(),
                    "domain_ground_true": q_json["domain"].strip(),
                }
            )

    if len(prompts) > 0:
        print("Example prompt:")
        print(prompts[0] if prompts else "No prompts generated.")

        responses = model.generate_batch_safe(
            prompts,
            do_sample=eval_config.do_sample,
            temperature=eval_config.temperature,
            max_new_tokens=eval_config.max_new_tokens,
            top_p=eval_config.top_p,
            top_k=eval_config.top_k,
            batch_size=eval_config.eval_batch_size,
        )

        # remove eos
        eos_token = model.tokenizer.eos_token
        cleaned_responses = [o.split(eos_token)[0].strip() for o in responses]
        

        # build ans_jsons by pairing question_jsons with cleaned_responses
        ans_jsons = []
        for prompt, resp in zip(question_jsons, cleaned_responses):
            ans_jsons.append(
                {
                    "questions": prompt["instruction"],
                    "response": resp.strip(),
                    "ground_true": prompt["model_name"].strip(),
                    "domain_ground_true": prompt["domain"].strip(),
                }
            )
    
    return ans_jsons

def create_dataframe(base_result_path: str, retriever_key: str | None = None, metric_type: str = "model") -> pd.DataFrame:
    """
    Create a dataframe for a specific accuracy metric.
    
    Args:
        base_result_path: Path to the results directory
        retriever_key: Name of the retriever (optional)
        metric_type: Type of metric - "model", "domain", or "family"
    """
    # Map metric type to JSON key and column suffix
    metric_mapping = {
        "model": ("Accuracy", "Accuracy (Before Snapping)", "M-Acc"),
        "domain": ("Accuracy Domain", "Accuracy Domain (Before Snapping)", "D-Acc"),
        "family": ("Accuracy Model Family", "Accuracy Model Family (Before Snapping)", "F-Acc")
    }
    
    json_key, json_key_before, col_suffix = metric_mapping[metric_type]
    
    rows = []

    for i, experience in enumerate(EXPERIENCES):
        row_name = f"Indexed(Exp {'+'.join([str(j + 1) for j in range(i + 1)])})"
        row_data = {"Configuration": row_name}

        # For each test set up to the current experience
        for j, test_set in enumerate(EXPERIENCES[: i + 1]):
            # Load metrics

            if len(RETRIEVERS) == 0:
                filename_metrics = os.path.join(
                    base_result_path,
                    f"exp-{experience}_test_set-{test_set}_metrics.json",
                )
            else:
                model_index_name = MODEL_INDICES[i]
                filename_metrics = os.path.join(
                    base_result_path,
                    f"exp-{experience}_model_index-{model_index_name}_test_set-{test_set}_retriever-{retriever_key}_metrics.json",
                )

            if os.path.exists(filename_metrics):
                with open(filename_metrics, "r") as f:
                    metrics = json.load(f)

                # Add both before snapping and regular accuracy metrics
                row_data[f"Experience {j + 1} {col_suffix} (Before)"] = round(
                    metrics.get(json_key_before, 0.0) * 100, 1
                )
                row_data[f"Experience {j + 1} {col_suffix}"] = round(
                    metrics.get(json_key, 0.0) * 100, 1
                )
            else:
                row_data[f"Experience {j + 1} {col_suffix} (Before)"] = None
                row_data[f"Experience {j + 1} {col_suffix}"] = None

        rows.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns to match the format: Configuration, Exp1 Acc (Before), Exp1 Acc, Exp2 Acc (Before), Exp2 Acc, ...
    column_order = ["Configuration"]
    for j in range(len(EXPERIENCES)):
        column_order.append(f"Experience {j + 1} {col_suffix} (Before)")
        column_order.append(f"Experience {j + 1} {col_suffix}")

    # Only keep columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Calculate Backward Transfer (BWT) before adding mean row (based on regular accuracy, not before snapping)
    num_experiences = len(EXPERIENCES)
    bwt_values = []
    
    for i in range(num_experiences):
        if i == 0:
            # First experience has no backward transfer
            bwt_values.append(None)
        else:
            # Calculate backward transfer for experience i+1 (1-indexed)
            # BWT = sum over j from 1 to i-1 of (A[i,j] - A[j,j])
            bwt = 0.0
            valid = True
            
            for j in range(i):
                # A[i,j]: row i (trained on exp 1..i+1), tested on exp j+1
                a_ij = df.iloc[i][f"Experience {j + 1} {col_suffix}"]
                # A[j,j]: row j (trained on exp 1..j+1), tested on exp j+1
                a_jj = df.iloc[j][f"Experience {j + 1} {col_suffix}"]
                
                if pd.isna(a_ij) or pd.isna(a_jj):
                    valid = False
                else:
                    bwt += (a_ij - a_jj)
            bwt /= i# Normalize by number of previous experiences
            bwt_values.append(round(bwt, 1) if valid else None)
    
    df[f"BWT {col_suffix}"] = bwt_values

    # Add mean row first (column-wise means)
    mean_row = {"Configuration": "Mean"}
    for col in column_order[1:]:  # Skip 'Configuration' column
        mean_val = df[col].mean()
        mean_row[col] = round(mean_val, 1) if pd.notna(mean_val) else None

    # Add BWT column to mean row
    bwt_mean = df[f"BWT {col_suffix}"].mean()
    mean_row[f"BWT {col_suffix}"] = round(bwt_mean, 1) if pd.notna(bwt_mean) else None
    
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Add Mean (Before) and Mean columns - row-wise means across experiences INCLUDING the mean row
    before_cols = [col for col in df.columns if "(Before)" in col and col_suffix in col]
    regular_cols = [col for col in df.columns if col_suffix in col and "(Before)" not in col and "BWT" not in col and "Mean" not in col]
    
    df[f"Mean (Before)"] = df[before_cols].mean(axis=1).apply(lambda x: round(x, 1) if pd.notna(x) else None)
    df[f"Mean"] = df[regular_cols].mean(axis=1).apply(lambda x: round(x, 1) if pd.notna(x) else None)

    return df
    


def create_snapping_dataframe(base_result_path: str, retriever_key: str | None = None) -> pd.DataFrame:
    """
    Create a dataframe for snapping-related metrics.
    For each experience, creates 3 columns: Accuracy_exist, Label_snapping_rate, Label_snapping_counter
    
    Args:
        base_result_path: Path to the results directory
        retriever_key: Name of the retriever (optional)
    """
    rows = []

    for i, experience in enumerate(EXPERIENCES):
        row_name = f"Indexed(Exp {'+'.join([str(j + 1) for j in range(i + 1)])})"  
        row_data = {"Configuration": row_name}

        # For each test set up to the current experience
        for j, test_set in enumerate(EXPERIENCES[: i + 1]):
            # Load metrics
            if len(RETRIEVERS) == 0:
                filename_metrics = os.path.join(
                    base_result_path,
                    f"exp-{experience}_test_set-{test_set}_metrics.json",
                )
            else:
                model_index_name = MODEL_INDICES[i]
                filename_metrics = os.path.join(
                    base_result_path,
                    f"exp-{experience}_model_index-{model_index_name}_test_set-{test_set}_retriever-{retriever_key}_metrics.json",
                )

            if os.path.exists(filename_metrics):
                with open(filename_metrics, "r") as f:
                    metrics = json.load(f)

                # Add the three metrics for this experience
                row_data[f"Accuracy_exist_exp{j + 1}"] = round(
                    metrics.get("Accuracy Exist", 0.0) * 100, 1
                )
                row_data[f"Label_snapping_counter_exp{j + 1}"] = metrics.get(
                    "Label Snapping Fixed", 0
                )
                row_data[f"Label_snapping_rate_exp{j + 1}"] = round(
                    metrics.get("Label Snapping Fix Rate", 0.0) * 100, 1
                )
            else:
                row_data[f"Accuracy_exist_exp{j + 1}"] = None
                row_data[f"Label_snapping_counter_exp{j + 1}"] = None
                row_data[f"Label_snapping_rate_exp{j + 1}"] = None

        rows.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns to match the format: Configuration, then for each exp: Accuracy_exist, Label_snapping_rate, Label_snapping_counter
    column_order = ["Configuration"]
    for j in range(len(EXPERIENCES)):
        column_order.append(f"Accuracy_exist_exp{j + 1}")
        column_order.append(f"Label_snapping_rate_exp{j + 1}")
        column_order.append(f"Label_snapping_counter_exp{j + 1}")

    # Only keep columns that exist in the dataframe
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Add mean row at the end
    mean_row = {"Configuration": "Mean"}
    for col in column_order[1:]:  # Skip 'Configuration' column
        if "counter" in col:
            # For counter, use sum instead of mean
            mean_val = df[col].sum()
        else:
            mean_val = df[col].mean()
        mean_row[col] = round(mean_val, 1) if pd.notna(mean_val) else None
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df


def create_summary_tables(base_result_path: str):
    """
    Create four separate summary tables for each retriever:
    1. Model Name Accuracy (M-Acc) with BWT
    2. Domain Accuracy (D-Acc) with BWT  
    3. Model Family Accuracy (F-Acc) with BWT
    4. Snapping Metrics (Accuracy Exist, Label Snapping Rate, Label Snapping Counter)
    
    Rows: Different model index name configurations (Indexed(Exp 1), Indexed(Exp 1-2), etc.)
    Columns: Accuracy for each experience + BWT (for 1-3), or multiple metrics per experience (for 4)
    """
    metric_types = ["model", "domain", "family"]
    metric_names = {
        "model": "model_accuracy",
        "domain": "domain_accuracy",
        "family": "model_family_accuracy"
    }
    
    if len(RETRIEVERS) == 0:
        for metric_type in metric_types:
            df = create_dataframe(
                base_result_path=base_result_path,
                metric_type=metric_type
            )
            summary_csv_filename = os.path.join(
                base_result_path, f"summary_{metric_names[metric_type]}.csv"
            )
            df.to_csv(summary_csv_filename, index=False)
            print(f"Saved {summary_csv_filename}")
        
        # Create snapping metrics summary
        df_snapping = create_snapping_dataframe(base_result_path=base_result_path)
        summary_csv_filename = os.path.join(
            base_result_path, "summary_snapping_metrics.csv"
        )
        df_snapping.to_csv(summary_csv_filename, index=False)
        print(f"Saved {summary_csv_filename}")
    else:
        for retriever_key in RETRIEVERS:
            for metric_type in metric_types:
                df = create_dataframe(
                    base_result_path=base_result_path,
                    retriever_key=retriever_key,
                    metric_type=metric_type
                )
                # save as CSV
                summary_csv_filename = os.path.join(
                    base_result_path, f"summary_{retriever_key}_{metric_names[metric_type]}.csv"
                )
                df.to_csv(summary_csv_filename, index=False)
                print(f"Saved {summary_csv_filename}")
            
            # Create snapping metrics summary for this retriever
            df_snapping = create_snapping_dataframe(
                base_result_path=base_result_path,
                retriever_key=retriever_key
            )
            summary_csv_filename = os.path.join(
                base_result_path, f"summary_{retriever_key}_snapping_metrics.csv"
            )
            df_snapping.to_csv(summary_csv_filename, index=False)
            print(f"Saved {summary_csv_filename}")
        

def write_metrics_and_answers(metrics: dict, answers: list, save_path: str):
    with open(save_path+"_answers.jsonl", "w") as f:
        for line in answers:
            f.write(json.dumps(line) + "\n")
    
    with open(save_path+"_metrics.json", "w") as f:
        json.dump(metrics, f)



def main():
    eval_config = EvalParser().parse_args()
    
    print("Evaluation Configuration:")
    print(eval_config)
    
    base_result_path = os.path.join("results", eval_config.variant_name)
    os.makedirs(base_result_path, exist_ok=True)

    global MODEL_INDICES, RETRIEVERS, EXPERIENCES, LORA_ADAPTERS_MAP, LORA_WEIGHTS
    if eval_config.model_indices:
        MODEL_INDICES = eval_config.model_indices
    if eval_config.retrievers:
        RETRIEVERS = eval_config.retrievers
    if eval_config.experience_names:
        EXPERIENCES = eval_config.experience_names
    if eval_config.lora_adapters:
        for i, exp_name in enumerate(EXPERIENCES):
            LORA_ADAPTERS_MAP[exp_name] = f"./core/experiments/{eval_config.lora_adapters[i]}" 
    if eval_config.lora_merging_strategy is not None:
        LORA_WEIGHTS = eval_config.weights   

    for i, experience in enumerate(EXPERIENCES):
        model_index_name = MODEL_INDICES[i] if len(MODEL_INDICES) > 0 else None

        # Build the cumulative model metadata for the current continual-learning step.
        # When evaluating experience k, domain/family credit should be computed against
        # the union of experiences 1..k, regardless of which earlier test set is used.
        cumulative_ground_truth_dataset = []
        for seen_experience in EXPERIENCES[: i + 1]:
            if seen_experience == "apibench":
                seen_dataset = ApibenchDataConfig()
            elif seen_experience == "mllm":
                seen_dataset = MLLMDataConfig()
            elif seen_experience == "hugging-bench-1":
                seen_dataset = HuggingBench1DataConfig()
            elif seen_experience == "hugging-bench-2":
                seen_dataset = HuggingBench2DataConfig()
            else:
                raise ValueError(f"Unknown experience name: {seen_experience}")

            cumulative_ground_truth_dataset.extend(
                load_dataset_json(seen_dataset.train_set)
            )

        for test_set in EXPERIENCES[: i + 1]:

            if test_set == "apibench":
                dataset = ApibenchDataConfig()
            elif test_set == "mllm":
                dataset = MLLMDataConfig()
            elif test_set == "hugging-bench-1":
                dataset = HuggingBench1DataConfig()
            elif test_set == "hugging-bench-2":
                dataset = HuggingBench2DataConfig()
            else:
                raise ValueError(f"Unknown experience name: {test_set}")

            dataset_json = load_dataset_json(dataset.test_set)
            model = None
            if eval_config.lora_adapters is not None and len(eval_config.lora_adapters) > 0:
                if eval_config.lora_merging_strategy is not None:
                    lora_paths = [f"./core/experiments/{adapter}" for adapter in eval_config.lora_adapters[: i + 1]]
                    eval_config.weights = LORA_WEIGHTS[: i + 1]
                    model = LoRAModelManager(eval_config, lora_paths=lora_paths)
                else:
                    model = LoRAModelManager(
                        eval_config, lora_paths=[LORA_ADAPTERS_MAP[experience]]
                    )

            if len(RETRIEVERS) > 0:
                for retriever_name in RETRIEVERS:
                    print(
                        f"Evaluating experience: {experience}, model_index: {model_index_name}, test_set: {test_set}, retriever: {retriever_name}"
                    )

                    answers = get_answers(
                        dataset_json,
                        model=model,
                        retriever_name=retriever_name,
                        model_index_name=model_index_name,
                        eval_config=eval_config,
                    )
                    metrics: dict = compute_metrics(
                        answers,
                        concatenated_ground_truth_dataset=cumulative_ground_truth_dataset,
                    )
                    
                    save_path = os.path.join(
                        base_result_path,
                        f"exp-{experience}_model_index-{model_index_name}_test_set-{test_set}_retriever-{retriever_name}",
                    )

                    write_metrics_and_answers(metrics, answers, save_path) 
                    
            else:
                print(
                    f"Evaluating experience: {experience}, test_set: {test_set}"
                )

                answers = get_answers(
                    dataset_json,
                    model=model,
                    retriever_name=None,
                    model_index_name=None,
                    eval_config=eval_config,
                )

                metrics: dict = compute_metrics(
                    answers,
                    concatenated_ground_truth_dataset=cumulative_ground_truth_dataset,
                )

                save_path = os.path.join(
                    base_result_path,
                    f"exp-{experience}_test_set-{test_set}",
                )
                write_metrics_and_answers(metrics, answers, save_path)

    create_summary_tables(base_result_path)


if __name__ == "__main__":
    main()
