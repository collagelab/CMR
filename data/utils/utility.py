import time
from typing import List
import pandas as pd
import json
import ast
import re
import os
import unicodedata
import Levenshtein

from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
import numpy as np

from huggingface_hub import ModelCard, HfApi

tqdm.pandas()

SLEEP_TIME = 0.4  # seconds



categories = {
    "Multimodal": [
        "Audio-Text-to-Text","Image-Text-to-Text","Image-to-Text", "Visual Question Answering",
        "Document Question Answering","Video-Text-to-Text","Visual Document Retrieval","Any-to-Any",
        "Feature Extraction", "Text-to-Video", "Text-to-Image", "Zero-Shot Image Classification",
        "Graph Machine Learning"
    ],
    "Computer Vision":[
        "Depth Estimation", "Image Classification", "Object Detection",
        "Image Segmentation", "Text-to-Image", "Image-to-Text", "Image-to-Image",
        "Image-to-Video", "Unconditional Image Generation","Video Classification",
        "Text-to-Video","Zero-Shot Image Classification", "Mask Generation",
        "Zero-Shot Object Detection", "Text-to-3D", "Image-to-3D",
        "Image Feature Extraction", "Keypoint Detection", "Video-to-Video"
    ],
    "Natural Language Processing": [
        "Text Classification","Token Classification","Table Question Answering",
        "Question Answering","Zero-Shot Classification","Translation","Summarization",
        "Feature Extraction","Text Generation","Fill-Mask",
        "Sentence Similarity","Text Ranking","Text2Text Generation", "Conversational"
    ],
    "Audio": [
        "Text-to-Speech", "Text-to-Audio", "Automatic Speech Recognition",
        "Audio-to-Audio", "Audio Classification", "Voice Activity Detection",
    ],
    "Tabular": [
        "Tabular Classification", "Tabular Regression","Time Series Forecasting",
    ],
    "Reinforcement Learning": [
        "Reinforcement Learning","Robotics",
    ],
    "Other": ["Graph Machine Learning"]
}


def get_closest_functionality(tags, functionalities_set, threshold=0.4):
  best_match = None
  min_dist = float('inf')

  for tag in tags:
    tag_cleaned = tag.replace("-", " ").lower()

    for func in functionalities_set:
        func_cleaned = func.replace("-", " ").lower()
        dist = Levenshtein.distance(tag_cleaned, func_cleaned)
        max_len = max(len(tag_cleaned), len(func_cleaned))
        norm_dist = dist / max_len if max_len > 0 else 1.0

        if norm_dist < min_dist:
            min_dist = norm_dist
            best_match = func

  if min_dist <= threshold:
      print(f"{best_match} assigned to tags: {tags}")
      return best_match
  else:
      raise ValueError(f"Could not determine functionality for tag(s): {tags}. The closest match found was '{best_match}', but the similarity was below the acceptable threshold.")
  

def get_model_dates(models, cutoff_date=pd.Timestamp("2023-05-25", tz="UTC")):
    model_to_date = {}
    api = HfApi()
    # use tqdm to show progress bar
    for model in tqdm(models, desc="Fetching model creation dates"):
        created = None
        try:
            model_info = api.model_info(model)
            # Safely get created_at if present; otherwise remain None
            created = getattr(model_info, "created_at", None)
            
            #if date is greater then cutoff_date set it to that date
            if created is not None and created > cutoff_date:
                created = cutoff_date
            
        except Exception as e:
            # failed to fetch model info; leave created as None
            # print(f"Warning: Could not fetch info for model {model}")
            # print(f"Error: {e}")
            created = None
        model_to_date[model] = created
        time.sleep(SLEEP_TIME)  # To avoid hitting API rate limits
    
    return model_to_date

def extract_model_name_from_api_call(api_call: str):  
    # 1. Try: --download_model model/name
    pattern_4 = r"--download_model\s+([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)"
    match = re.search(pattern_4, api_call)
    if match:
        return match.group(1)


    # 2. Try Hugging Face hub style: "username/model-name"
    pattern_1 = r'([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)'
    match = re.search(pattern_1, api_call)
    if match:
        return match.group(1)

    # 3. Try: model='model-name' or from_pretrained('model-name')
    pattern_2 = r"(?:model=|from_pretrained\()\s*['\"]([A-Za-z0-9._-]+)['\"]"
    match = re.search(pattern_2, api_call)
    if match:
        return match.group(1)

    # 4. Try: timm.create_model('model-name', ...)
    pattern_3 = r"timm\.create_model\(\s*['\"]([A-Za-z0-9._-]+)['\"]"
    match = re.search(pattern_3, api_call)
    if match:
        return f"timm/{match.group(1)}"

    if "RandomForestRegressor" in api_call or "WhisperModel" in api_call:
      return api_call

    return None


def remove_fields_from_api_data(d: dict):
    if not isinstance(d, dict):
        raise ValueError("Input is not a dictionary", d)

    fields_to_remove = ['api_arguments', 'python_environment_requirements', 'example_code']
    return {k: v for k, v in d.items() if k not in fields_to_remove}

def inject_model_name(api, model):
    if isinstance(api, dict):
        api = dict(api)
        api['model_name'] = model
        return api
    return api

def extract_instruction_apibench(text: str):
    # Check if text is a string, if it not a string return nan
    if not isinstance(text, str):
        return pd.NA

    # If it is a string, splits by "###Output" and return the first occurrence
    if "### Output" in text:
      text = text.replace("### Output", "###Output")

    if "#### Output" in text:
      text = text.replace("#### Output", "###Output")

    text = text.split("###Output")[0].strip()

    if "###output:" in text.lower() or "### output:" in text.lower():
      print(text)
      raise ValueError("output is in the text")


    return text


def load_dataframe(file_path: str, lines: True) -> pd.DataFrame:
    """Load a JSON file and return a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_json(file_path, lines=lines)
  
  
def extract_repo_model_name_torchhub(row):
    repo_match = re.search(r"repo_or_dir\s*=\s*['\"]([^'\"]+)['\"]", row)
    model_match = re.search(r"model\s*=\s*['\"]([^'\"]+)['\"]", row)

    if repo_match:
        repo = repo_match.group(1)

        if ":" in repo:
          repo = repo.split(":")[0]

        if model_match:
            model = model_match.group(1)
            return repo, model
        return repo, None
    return None, None
  

def extract_instruction_torchhub(text):
  # Check if text is a string, if it not a string return nan
  if not isinstance(text, str):
      return pd.NA

  # If it is a string, splits by "###Output" and return the first occurrence
  if "###Output" in text:
    return text.split("###Output")[0].strip()

  try:
    # Case 1: Markdown style with ###Instruction
    # if "###Instruction:" in row:
    #   return (
    #       row.split("###Instruction:")[1]
    #           .split("###Output:")[0]
    #           .strip()
    #   )

    # Case 2: Pseudo-dict string
    match = re.search(
        r"[\"']?Instruction[\"']?\s*:\s*(.*?)(?=,\s*[\"']?Output[\"']?\s*:)",
        text, re.DOTALL
    )
    if match:
        instruction = match.group(1).strip()
        # Remove surrounding quotes if present
        if instruction.startswith(("'", '"')) and instruction.endswith(("'", '"')):
            instruction = instruction[1:-1]
        return "###Instruction:"+instruction
    else:
        print(f"No match found for row: {text}")

  except Exception as e:
      print(f"Error processing data: {e}")
      return None


def combine_repo_model(repo, model):
    if pd.notna(repo) and pd.notna(model):
        return f"{repo}/{model}"
    elif pd.notna(repo):
        return repo
    elif pd.notna(model):
        return model
    else:
        return None
      
      
def get_explanation_torchhub(text):
  # Pattern to match 'explanation': followed by the content in quotes
  pattern = r"'explanation':\s*[\"']([^\"']*)[\"']"
  match = re.search(pattern, text)
  if match:
      return match.group(1)
  return ""


def clean_markdown(markdown_text):
    if markdown_text is None:
        return ""

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)

    # Remove HTML tags <...>
    text = re.sub(r"<[^>]+>", "", text)

    # Remove images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)

    # Remove links [text](url)
    text = re.sub(r"\[[^\]]*\]\([^\)]*\)", "", text)

    # Remove direct URLs (http://, https://)
    text = re.sub(r"https?://\S+", "", text)

    return text


def wellness_card(markdown_text):
    if markdown_text is None:
        return False

    markdown_text = clean_markdown(markdown_text)
    lower_md = markdown_text.lower()

    required_sections = ["description", "introduction", "summary", "details"]  # le parole chiave da cercare in ordine
    matches = [s for s in required_sections if s in lower_md]
    return len(matches) >= 1


def extract_description(markdown_text):
    if not markdown_text:
        return None

    if "this model card has been automatically generated" in markdown_text.lower():
        return None

    markdown_text = clean_markdown(markdown_text)
    # Rimuove blocchi di codice
    text_no_code = re.sub(r"```.*?```", "", markdown_text, flags=re.DOTALL)

    # Split in linee
    lines = [l.rstrip() for l in text_no_code.splitlines()]

    output_paragraphs = []

    # 1. Sempre aggiungi le prime 2–3 linee significative (ignora headers o "Tags:")
    first_lines = []
    for line in lines:
        if not line.startswith("#") and not line.lower().startswith("tags:") and line.strip():
            first_lines.append(line.strip())
        if len(first_lines) >= 3:
            break
    if first_lines:
        output_paragraphs.append(" ".join(first_lines))

    # 2. Cerca header con una keyword
    keywords = ["model description", "description", "introduction", "summary","key features"]
    capture = False
    captured_lines = []

    for line in lines:
        header_match = re.match(r"^(#+)\s*(.*)$", line)
        if header_match:
            header_text = header_match.group(2).strip().lower()
            if any(header_text.startswith(k.lower()) for k in keywords) and not capture:
                # Trovata la prima keyword
                capture = True
                continue
            elif capture:
                # Inizio di un nuovo header dopo la keyword → fermati
                break
        if capture and line.strip():
            captured_lines.append(line.strip())

    if captured_lines:
        output_paragraphs.append("\n".join(captured_lines))

    return "\n\n".join(output_paragraphs) if output_paragraphs else None

def normalize_text(text, remove_numbers=False, remove_cjk=True):
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Normalize unicode characters (e.g., accented letters)
    text = unicodedata.normalize("NFKC", text)

    # Remove common formatting and markup symbols
    text = re.sub(r"[•●▪♦▪■□◆–—−]", " ", text)  # bullet points
    text = re.sub(r"[|=_~*•<>#]", " ", text)     # markup symbols
    text = re.sub(r"[\[\]{}()<>]", "", text)     # brackets

    # Remove numbers (optional)
    if remove_numbers:
        text = re.sub(r"\b\d+(\.\d+)?\b", "", text)

    # Remove CJK characters (Chinese, Japanese, Korean)
    if remove_cjk:
        # Remove Unicode blocks for CJK
        text = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+", "", text)

    # Remove multiple spaces and newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"-{2,}", "", text)

    # Remove trailing punctuation and whitespace
    text = text.strip()

    return text

def get_modelCardCleaned(markdown_text: str):
  markdown_text = extract_description(markdown_text)
  markdown_text = normalize_text(markdown_text)
  # if text contains more than 20 words return text
  if len(markdown_text.split()) > 20:
    return markdown_text
  else:
    return None



def get_description_from_model_card(model_card: ModelCard):
    res = get_modelCardCleaned(model_card.text)
    if res is None:
      try:
          match = re.search(r"# .*?\n\n(.*?)(?:\n\n## |\Z)", model_card.text, re.DOTALL)
          if match:
              return match.group(1).strip()
      except Exception:
          return None
    else:
      return res
  
CUSTOM_DOMAIN_MAPPING = {
    "Semantic Segmentation": "Computer Vision Image Segmentation",
    "Classification": "Tabular Tabular Classification",
    "Text-to-Speech": "Audio Text-to-Speech",
    "Text-To-Speech": "Audio Text-to-Speech",
    "Audio Separation": "Audio Audio-to-Audio",
    "Object Detection": "Computer Vision Object Detection",
    "Video Classification": "Computer Vision Video Classification",
    "Video classification": "Computer Vision Video Classification",
    
    "Audio event classification": "Audio Audio Classification",
    "Image classification": "Computer Vision Image Classification",
    "Text preprocessing": "Natural Language Processing Feature Extraction",     # suggested map, not sure if correct
    "Text classification": "Natural Language Processing Text Classification",
    "Image Frame Interpolation": "Computer Vision Image-to-Image",
    "Audio Speech-to-Text": "Audio Automatic Speech Recognition",
    "Text language model": "Natural Language Processing Text Generation",
    "Text embedding": "Natural Language Processing Feature Extraction",
    "Image pose detection": "Computer Vision Object Detection",
    "Image segmentation": "Computer Vision Image Segmentation",
    "Audio embedding": "Audio Audio Classification",
    "Image object detection": "Computer Vision Object Detection",
}

def update_api_data_domain(api_data, domain_mapping):        
        # Determine original type to preserve it when returning
        original_is_dict = isinstance(api_data, dict)

        parsed = None
        if original_is_dict:
            parsed = api_data.copy()
        elif isinstance(api_data, str):
            # try JSON first, then ast literal eval
            try:
                parsed = json.loads(api_data)
            except Exception:
                try:
                    parsed = ast.literal_eval(api_data)
                except Exception:
                    raise ValueError("api_data string is neither valid JSON nor a Python literal", api_data)
                
        else:
            raise ValueError("Unsupported api_data type", type(api_data), api_data)

        if isinstance(parsed, dict):
            old_domain = parsed.get("domain")
            new_domain = domain_mapping.get(old_domain, old_domain)
            parsed["domain"] = new_domain

        # Return same type as input: dict -> dict, str -> JSON string
        if original_is_dict:
            return parsed
        else:
            try:
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return api_data
        
def unify_domains(df: pd.DataFrame) -> pd.DataFrame:    
    other_domains = list(set(df["domain"].unique().tolist()))
    
    df = df.copy()
    # build mapping dynamically — if label not in dictionary, keep original
    domain_mapping = {
        domain: CUSTOM_DOMAIN_MAPPING.get(domain, domain)
        for domain in other_domains
    }
    df.loc[:, "domain"] = df["domain"].map(domain_mapping)

    # Also update the `domain` value inside the `api_data` field when present.
    df.loc[:, "api_data"] = df["api_data"].apply(lambda x: update_api_data_domain(x, domain_mapping))

    return df

def normalize_model_name(name: str) -> str:
    if name is None or name == "":
        raise ValueError("Model name is None or empty")
        
    SIZE_WORDS = {"tiny","mini","small","base","medium","large","xl","xxl","xxxl"}
    TASK_WORDS = {"finetuned","fine-tuned","ft","sft","instruct","chat"}  # optional: treat separately
    VERSION_RE = re.compile(r"^v\d+(\.\d+)?(_\d+)?$")                   # v2, v1.1, v1_1
    PARAM_RE   = re.compile(r"^\d+(\.\d+)?b(\d+)?$") 
    
    raw = name.lower().split("/")[-1]
    raw = raw.replace("_", "-")
    raw = re.sub(r"[^\w\-\.]+", "-", raw)      # drop odd punctuation, keep dots if you want
    raw = re.sub(r"-+", "-", raw).strip("-")

    toks: List[str] = [t for t in raw.split("-") if t]

    # Remove version / param-count tokens only if they are standalone tokens
    kept = []
    for t in toks:
        if VERSION_RE.match(t):
            continue
        if PARAM_RE.match(t):
            continue
        kept.append(t)

    # Remove size words only if they are standalone tokens
    kept2 = [t for t in kept if t not in SIZE_WORDS]

    # If you removed everything, fall back to the less-aggressive version
    final = kept2 if kept2 else kept
    out = "-".join(final).strip("-")

    if not out:
        raise ValueError(f"Model name became empty - original: {name}")
    return out


def root_token(norm_name: str) -> str:
    t0 = norm_name.split("-")[0]
    # Optional canonicalization rules
    if t0 in {"distilbert"}: return "bert"
    if t0 in {"xlm"}: return "xlm-roberta"  # if you normalize that way
    return t0

def get_family_map(df: pd.DataFrame) -> dict:
    names = df["normalized_model_name"].unique().tolist()

    # Block by root token
    buckets = defaultdict(list)
    for n in names:
        buckets[root_token(n)].append(n)

    family_map = {}
    next_label = 0

    for _, bucket_names in buckets.items():
        if len(bucket_names) == 1:
            family_map[bucket_names[0]] = next_label
            next_label += 1
            continue

        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
        X = vec.fit_transform(bucket_names)
        dist = cosine_distances(X)

        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=0.7,
            n_clusters=None,
        )
        labels = clustering.fit_predict(dist)

        # Re-label to keep global uniqueness
        label_map = {l: (i + next_label) for i, l in enumerate(sorted(set(labels)))}
        for n, l in zip(bucket_names, labels):
            family_map[n] = label_map[l]
        next_label += len(label_map)


    return family_map


def map_to_train_families(eval_df, train_family_map, family_name_map):
    """
    Map eval set models to the closest family from train set.
    
    Args:
        eval_df: Evaluation dataframe with normalized_model_name column
        train_family_map: Dictionary mapping train normalized names to family IDs
        family_name_map: Dictionary mapping family IDs to representative family names
    
    Returns:
        eval_df with family_id and model_family columns added
    """
    train_normalized_names = list(train_family_map.keys())
    eval_df = eval_df.copy()
    
    def find_closest_family(eval_name):
        """Find the closest family for a given eval model name."""
        min_dist = float('inf')
        closest_train_name = None
        
        for train_name in train_normalized_names:
            dist = Levenshtein.distance(eval_name, train_name)
            norm_dist = dist / max(len(eval_name), len(train_name))
            
            if norm_dist < min_dist:
                min_dist = norm_dist
                closest_train_name = train_name
        
        # Always assign to the closest family found
        return train_family_map[closest_train_name], min_dist
    
    # Map each eval model to closest train family
    eval_df[['family_id', 'family_distance']] = eval_df['normalized_model_name'].apply(
        lambda x: pd.Series(find_closest_family(x))
    )
    
    # Map family IDs to family names
    eval_df['model_family'] = eval_df['family_id'].map(family_name_map)
    
    return eval_df


def add_model_family(dfs: list[pd.DataFrame], previous_train_df: pd.DataFrame | None = None) -> list[pd.DataFrame]:
    """
    Compute model families for continual learning: combine previous experience with current experience.
    
    Args:
        dfs: List containing [train_df, eval_df] for current experience with normalized_model_name column
        previous_train_df: Training dataframe from previous experience(s) with normalized_model_name column
    
    Returns:
        List containing [train_df, eval_df] with family_id and model_family columns added
    """
    train_df = dfs[0].copy()
    eval_df = dfs[1].copy()
    
    # Combine previous and current train normalized names
    if previous_train_df is not None:
        # Ensure previous_train_df has normalized_model_name column
        if 'normalized_model_name' not in previous_train_df.columns:
            previous_train_df = previous_train_df.copy()
            previous_train_df['normalized_model_name'] = previous_train_df['model_name'].apply(normalize_model_name)
        
        combined_df = pd.concat([
            previous_train_df[['normalized_model_name', 'model_name']],
            train_df[['normalized_model_name', 'model_name']]
        ], ignore_index=True).drop_duplicates(subset=['model_name'])
    else:
        combined_df = train_df[['normalized_model_name', 'model_name']].drop_duplicates(subset=['model_name'])
    
    # Compute families on COMBINED set (previous + current train) using normalized names
    combined_family_map = get_family_map(combined_df)
    
    # Create a mapping from model_name to family_id (ensuring one model = one family)
    model_name_to_family = {}
    for _, row in combined_df.iterrows():
        norm_name = row['normalized_model_name']
        orig_name = row['model_name']
        if norm_name in combined_family_map:
            model_name_to_family[orig_name] = combined_family_map[norm_name]
    
    # Compute family representative names based on combined clustering
    family_name_map = (
        combined_df.assign(family_id=combined_df['normalized_model_name'].map(combined_family_map))
        .groupby('family_id')['normalized_model_name']
        .agg(lambda x: x.value_counts().idxmax())
    )
    
    # Map current train using model_name -> family_id mapping
    train_df["family_id"] = train_df["model_name"].map(model_name_to_family)
    train_df["model_family"] = train_df["family_id"].map(family_name_map)
    
    # Map eval using model_name -> family_id mapping
    # For new models not in combined_df, find closest normalized name
    def assign_family(row):
        if row['model_name'] in model_name_to_family:
            return model_name_to_family[row['model_name']]
        else:
            # New model: find closest normalized name from training set
            norm_name = row['normalized_model_name']
            min_dist = float('inf')
            closest_norm = None
            for train_norm in combined_family_map.keys():
                dist = Levenshtein.distance(norm_name, train_norm)
                norm_dist = dist / max(len(norm_name), len(train_norm))
                if norm_dist < min_dist:
                    min_dist = norm_dist
                    closest_norm = train_norm
            return combined_family_map.get(closest_norm) if closest_norm else None
    
    eval_df["family_id"] = eval_df.apply(assign_family, axis=1)
    eval_df["model_family"] = eval_df["family_id"].map(family_name_map)
    
    return [train_df, eval_df]