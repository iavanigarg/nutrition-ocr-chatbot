import os
import requests
import logging
import time
import string

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

CLASSIFIER_PROMPT = """
You are an intent classifier.

Classify every user query into exactly one category.

GENERAL:
The question is clearly independent of uploaded documents and can be answered using general knowledge.

DOCUMENT:
The question asks for information that could reasonably exist inside one or more uploaded documents, including summaries, names, tables, values, architecture, metadata, comparisons, or any document-specific information.

Important:
If there is any uncertainty, always classify as DOCUMENT.

Return exactly one word:

GENERAL

or

DOCUMENT

Do not explain your answer.
Do not add punctuation.
Do not add markdown.
Do not output anything except one word.
"""

def classify_query(question: str) -> str:
    """
    Classifies a user query into 'GENERAL' or 'DOCUMENT' using a lightweight zero-shot LLM call.
    Returns 'DOCUMENT' as a safe fallback on failure.
    """
    if not question or not question.strip():
        return "GENERAL"
        
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "system": CLASSIFIER_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 10
        }
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        raw_answer = data.get("response", "").strip().upper()
        
        logger.info(f"Classifier model: {MODEL_NAME}")
        logger.info(f"Classifier question: {question}")
        logger.info(f"Classifier raw response: {raw_answer}")
        
        # Normalize the model output before returning
        normalized_answer = raw_answer.strip(string.punctuation).strip()
        
        if normalized_answer == "GENERAL":
            intent = "GENERAL"
        else:
            # Safely fall back to DOCUMENT if unexpected or cannot be confidently parsed
            intent = "DOCUMENT"
            
        logger.info(f"Classifier final intent: {intent}")
        return intent
        
    except Exception as e:
        logger.error(f"Failed to classify query: {e}. Falling back to DOCUMENT routing.")
        return "DOCUMENT"
