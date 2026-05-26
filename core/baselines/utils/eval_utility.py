import Levenshtein


# Function to compute Levenshtein similarity
def levenshtein_similarity(name1, name2):
    max_len = max(len(name1), len(name2))
    if max_len == 0:
        return 1.0  # If both strings are empty, consider them identical
    return 1 - (Levenshtein.distance(name1, name2) / max_len)


def validate_and_fix_response(
    response, model_names, model_domains=None, ground_truth_domain=None, threshold=0.6
):
    """
    Label snapping: Validate response and fix if not in model names.

    This function implements post-processing validation to fix hallucinated
    model names by finding the closest valid model name using Levenshtein similarity.
    If a ground truth domain is provided, it prioritizes models from that domain.

    Args:
        response: The generated response (model name)
        model_names: Set of valid model names
        model_domains: Dict mapping model names to domains (optional)
        ground_truth_domain: Expected domain (optional, for domain filtering)
        threshold: Minimum similarity threshold for fixing (default: 0.6)

    Returns:
        Fixed response (original if no good match found)
    """
    response = response.strip()

    # If already valid, return as-is
    if response in model_names:
        return response

    # Find closest match
    best_match = None
    best_sim = 0

    # Filter by domain if provided (prioritize correct domain)
    candidate_models = model_names
    if ground_truth_domain and model_domains:
        domain_models = {
            m for m in model_names if model_domains.get(m) == ground_truth_domain
        }
        # If we have domain matches, use those; otherwise fall back to all models
        if domain_models:
            candidate_models = domain_models

    # Find best match by Levenshtein similarity
    for model_name in candidate_models:
        sim = levenshtein_similarity(response, model_name)
        if sim > best_sim:
            best_sim = sim
            best_match = model_name

    # Return best match if similarity is above threshold
    if best_match and best_sim >= threshold:
        return best_match

    # Return original response if no good match found
    return response


def compute_metrics(
    answers, concatenated_ground_truth_dataset, enable_label_snapping=True, snapping_threshold=0.8
):
    """
    Compute evaluation metrics with optional label snapping.

    Args:
        answers: List of answer dictionaries
        concatenated_ground_truth_dataset: Dataset with model names and domains
        enable_label_snapping: Whether to enable post-processing validation (default: True)
        snapping_threshold: Similarity threshold for label snapping (default: 0.7)
    """
    model_domains = {
        data["model_name"]: data["domain"] for data in concatenated_ground_truth_dataset
    }
    model_names = {data["model_name"] for data in concatenated_ground_truth_dataset}
    model_families = {
        data["model_name"]: data["model_family"] for data in concatenated_ground_truth_dataset
    }
    
    
    count_exist_before_snapping = 0
    count_domain_before_snapping = 0
    count_model_family_before_snapping = 0
    count_before_snapping = 0
    
    count_exist = 0
    
    same_domain = 0
    count = 0
    model_family_counter = 0
    fixed_count = 0  # Track how many responses were fixed by label snapping

    for ans in answers:
        original_response = ans["response"]
        ground_true_model = ans["ground_true"]
        ground_truth_domain = ans.get("domain_ground_true")
        ground_truth_family = model_families.get(ground_true_model)

        # Apply label snapping if enabled
        if enable_label_snapping:
            ans["response"] = validate_and_fix_response(
                ans["response"],
                model_names,
                model_domains,
                ground_truth_domain,
                snapping_threshold,
            )
            # Track if response was fixed
            if ans["response"] != original_response:
                fixed_count += 1

        # Compute accuracy metrics
        if ans["response"] == ans["ground_true"]:
            count += 1
            
        if original_response == ans["ground_true"]:
            count_before_snapping += 1

        if ans["response"] in model_names:
            count_exist += 1
            predicted_domain = model_domains.get(ans["response"])
            predicted_family = model_families.get(ans["response"])

            if predicted_domain is not None and predicted_domain == ground_truth_domain:
                same_domain += 1
            
            if predicted_family is not None and predicted_family == ground_truth_family:
                model_family_counter += 1
        
        if original_response in model_names:
            count_exist_before_snapping += 1
            original_domain = model_domains.get(original_response)
            original_family = model_families.get(original_response)

            if original_domain is not None and original_domain == ground_truth_domain:
                count_domain_before_snapping += 1
            
            if original_family is not None and original_family == ground_truth_family:
                count_model_family_before_snapping += 1

    accuracy = count / len(answers)
    accuracy_exist = count_exist / len(answers)
    accuracy_domain = same_domain / len(answers)
    accuracy_model_family = model_family_counter / len(answers)
    
    accuracy_before_snapping = count_before_snapping / len(answers)
    accuracy_exist_before_snapping = count_exist_before_snapping / len(answers)
    accuracy_domain_before_snapping = count_domain_before_snapping / len(answers)
    accuracy_model_family_before_snapping = count_model_family_before_snapping / len(answers)

    metrics = {
        "Accuracy (Before Snapping)": accuracy_before_snapping,
        "Accuracy": accuracy,
        "Accuracy Exist (Before Snapping)": accuracy_exist_before_snapping,
        "Accuracy Exist": accuracy_exist,
        "Accuracy Domain (Before Snapping)": accuracy_domain_before_snapping,
        "Accuracy Domain": accuracy_domain,
        "Accuracy Model Family (Before Snapping)": accuracy_model_family_before_snapping,
        "Accuracy Model Family": accuracy_model_family,
    }

    # Add label snapping statistics if enabled
    if enable_label_snapping:
        metrics["Label Snapping Fixed"] = fixed_count
        metrics["Label Snapping Fix Rate"] = (
            fixed_count / len(answers) if len(answers) > 0 else 0.0
        )

    return metrics
