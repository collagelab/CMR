import pandas as pd
import argparse
import json
import ast


def calculate_retriever_accuracy(df):
    """Calculate accuracy for each retriever column in the dataframe"""
    retriever_columns = [col for col in df.columns if "retrieved_info" in col]
    results = {}

    for col in retriever_columns:
        correct_predictions = 0
        correct_domains = 0

        for idx, row in df.iterrows():
            try:
                # Get the model_name (ground truth)
                ground_truth_api_name = row["api_data"]["api_name"]
                ground_truth_domain = row["domain"]

                retrieved_info_str = row[col]
                retrieved_info = ast.literal_eval(retrieved_info_str)


                # Check name accuracy
                if "api_name" not in list(retrieved_info.keys()) or "domain" not in list(retrieved_info.keys()):
                    raise KeyError(f"'api_name' key or 'domain' key not found in retrieved_info for row {idx}")
                                
                if retrieved_info["api_name"] == ground_truth_api_name:
                    correct_predictions += 1
                
                if retrieved_info["domain"] == ground_truth_domain:
                    correct_domains += 1

            except (ValueError, KeyError, TypeError, SyntaxError) as e:
                print(
                    f"Row {idx}: Error parsing JSON or missing key in column {col}\nRow: {row}\n"
                )
                raise e

        model_name_accuracy = correct_predictions / len(df)
        domain_accuracy = correct_domains / len(df)

        results[col] = {
            "model_name_accuracy": model_name_accuracy,
            "correct_names": correct_predictions,
            "domain_accuracy": domain_accuracy,
            "correct_domains": correct_domains,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Available data options:
                apibench
                mllm

                Examples:
                python retrievers_accuracy.py --data apibench 
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["apibench", "mllm"],
        help="Specify which dataset to process",
    )

    args = parser.parse_args()

    if args.data == "apibench":
        df_train = pd.read_json("../data/processed/cleaned-apibench-all-train.json", lines=True)
        df_val = pd.read_json("../data/processed/cleaned-apibench-all-val.json", lines=True)
        df_eval = pd.read_json("../data/processed/cleaned-apibench-all-eval.json", lines=True)

    elif args.data == "mllm":
        df_train = pd.read_json("../data/processed/cleaned-mllm-train.json", lines=True)
        df_val = pd.read_json("../data/processed/cleaned-mllm-val.json", lines=True)
        df_eval = pd.read_json("../data/processed/cleaned-mllm-eval.json", lines=True)

    df_train = pd.concat([df_train, df_val], ignore_index=True)

    train_results = calculate_retriever_accuracy(df_train)
    eval_results = calculate_retriever_accuracy(df_eval)

    all_retrievers = set(train_results.keys()) | set(eval_results.keys())

    for retriever in sorted(all_retrievers):
        print(f"\n{retriever}:")
        if retriever in train_results:
            train_model_acc = train_results[retriever]["model_name_accuracy"]
            train_domain_acc = train_results[retriever]["domain_accuracy"]
            train_correct_names = train_results[retriever]["correct_names"]
            train_correct_domains = train_results[retriever]["correct_domains"]
            print(f"  Training Model Name Accuracy: {train_model_acc:.4f} ({train_correct_names}/{len(df_train)})")
            print(f"  Training Domain Accuracy: {train_domain_acc:.4f} ({train_correct_domains}/{len(df_train)})")
        if retriever in eval_results:
            eval_model_acc = eval_results[retriever]["model_name_accuracy"]
            eval_domain_acc = eval_results[retriever]["domain_accuracy"]
            eval_correct_names = eval_results[retriever]["correct_names"]
            eval_correct_domains = eval_results[retriever]["correct_domains"]
            print(f"  Evaluation Model Name Accuracy: {eval_model_acc:.4f} ({eval_correct_names}/{len(df_eval)})")
            print(f"  Evaluation Domain Accuracy: {eval_domain_acc:.4f} ({eval_correct_domains}/{len(df_eval)})")

    # Save results to JSON file
    output_file = f"../results/retriever_accuracy_results_{args.data}.json"
    results_summary = {
        "training_results": train_results,
        "evaluation_results": eval_results,
    }

    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
