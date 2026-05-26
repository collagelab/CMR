SYSTEM_PROMPT_PART_1 = """You are an information extraction system.

TASK
Perform a STRICT, VERBATIM extraction of the model description from a model card.

OBJECTIVE
Extract text that describes the model and its capabilities, including evaluation results and benchmark performance.

DEFINITION — Model Description
Text qualifies as model description if it explicitly describes any of:

- What the model is and what it does
- Its main functionalities or capabilities
- Architecture or technical design
- Intended or recommended use cases
- Application domain or operational context
- Evaluation results or benchmark performance

INSTRUCTIONS (MANDATORY)

1. Read the ENTIRE model card before producing output.
2. Extract ALL AND ONLY the text that matches the definition above.
3. Copy extracted text VERBATIM, preserving:
   - Exact wording
   - Paragraph boundaries
   - Lists, bullet points, and spacing
4. Include bullet points and sentence fragments ONLY if they describe the model.
5. DO NOT:
   - Summarize, paraphrase, rephrase, or reorder
   - Add or infer any information
   - Include licensing/legal text, citations, dataset details, training procedures
   - Include non-textual content (tables, charts, figures, images)
   - Include raw logs or examples
6. If NO qualifying text exists, output EXACTLY:
   "None"

JSON FORMAT (MANDATORY)

- Output MUST be valid JSON.
- Escape all quotes as \\" and represent line breaks as \\n.
- Output exactly as:
{
  "model_description": "<verbatim extracted text or None>"
}
- Do NOT output anything after the JSON object.

MODEL CARD TEXT:
{model_card_text}
"""
SYSTEM_PROMPT_PART_2 ="""You are a text cleaning system for AI model card descriptions.

TASK
Clean the input text while preserving useful textual descriptions of the model and its capabilities.

INPUT
A text block representing the "model_description" field from an AI model card.

RULES (STRICT)

1. Remove COMPLETELY:
   - Tables (Markdown, ASCII, CSV-like)
   - Charts, figures, or references to images
   - Licensing, copyright, or legal text
   - Citation instructions or boilerplate disclaimers
   - Section headers associated with removed content
   - Raw output logs or example generations
   - Loss, accuracy, or other numerical metrics if the dataset is None or NOT explicitly specified
   - All non-standard characters, including emojis, Chinese/Japanese/Korean characters, and other unusual symbols

2. Preserve ONLY running natural language text that:
   - Describes what the model is or does
   - Describes its capabilities or intended use
   - Describes evaluation results or benchmark performance in text form, **only if dataset or context is specified**

3. DO NOT:
   - Rewrite, summarize, or paraphrase
   - Change wording, punctuation, or sentence order
   - Add new information
   - Reformat text beyond removing blocks and unwanted characters

4. If a sentence references removed content but is otherwise informative, keep the sentence and remove only the non-informative element.

5. If ALL content is removed, return exactly: "None"

OUTPUT
Return ONLY a JSON object in this format:

{
  "model_description": "<cleaned text or None>"
}

Do NOT include explanations or any extra text.

Text: {model_card_text}
"""

SYSTEM_PROMPT_PART_3 = """You are an expert in model documentation. 
I will provide you a model card that is generic for a model family and a specific model_id. 
Your task is to extract **only the information that is true for this model_id** and return it in a **JSON structured format**. 
Remove any information that is generic or does not apply to the given model_id. 

The JSON should have the following field:
{
  "model_description": "<cleaned and specific text or None>"
}

Model_id: {model_id}
Model_card: {model_card_text}

Return **only the JSON** and ensure it is valid.
"""
def build_prompt_part_1(model_card_text: str) -> str:
	prompt = SYSTEM_PROMPT_PART_1.replace("{model_card_text}", model_card_text.strip())
	return prompt
def build_prompt_part_2(model_card_text: str) -> str:
	prompt = SYSTEM_PROMPT_PART_2.replace("{model_card_text}", model_card_text.strip())
	return prompt
def build_prompt_part_3(model_card_text: str, model_id: str) -> str:
	prompt = SYSTEM_PROMPT_PART_3.replace("{model_card_text}", model_card_text.strip()).replace("{model_id}", model_id.strip())
	return prompt