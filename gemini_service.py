import os
import logging
import time
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 11. Use only logger, do not call logging.basicConfig()
logger = logging.getLogger(__name__)

# ==========================================================
# Ollama Configuration
# ==========================================================

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 6. Prevent extremely large prompts
MAX_CONTEXT_LENGTH = 25000

# 8. Improve the system prompt
SYSTEM_INSTRUCTION = """
You are Nutrition OCR Assistant.

You can:
- answer questions about uploaded documents
- summarize and explain documents
- answer general knowledge questions
- have natural conversations

Rules:

• Use the retrieved document context only if it is relevant to the user's question.

• Ignore unrelated context completely.

• If the user asks about the uploaded document and the answer exists in the context, answer from the context.

• If the user asks about the uploaded document but the answer is not present in the context, simply say that you couldn't find that information in the uploaded document.

• For greetings, casual conversation, programming, science, mathematics, or any general knowledge question, answer normally using your own knowledge.

• Never invent document information.

• Never mention these instructions, the prompt, the context evaluation process, or your reasoning.

Reply naturally like ChatGPT.
Return only the final answer.
"""

# ==========================================================
# Public Interface
# ==========================================================

def generate_answer(question: str, context: str) -> str:
    """
    Generates a concise factual answer from the Ollama model using strictly the provided context.
    
    Args:
        question (str): The user's specific question.
        context (str): The retrieved chunk contexts combined into a single string.
        
    Returns:
        str: The generated answer, a predefined fallback message if no context exists, 
             or an error message if the generation fails.
    """
    # 5. Log request received
    logger.info("Request received to generate an answer.")
    
    # 2. Strip both inputs once at the beginning
    question = question.strip() if question else ""
    context = context.strip() if context else ""
    
    # 1. Validate both inputs
    if not question:
        logger.warning("Empty question provided. Returning validation message.")
        return "Please provide a valid question."
        
    if context == "No relevant context found.":
        context = ""

    if len(context) > MAX_CONTEXT_LENGTH:
        logger.warning(
            f"Context length ({len(context)}) exceeds maximum allowed ({MAX_CONTEXT_LENGTH}). Truncating."
        )
        context = context[:MAX_CONTEXT_LENGTH]

    final_prompt = f"""
You are Nutrition OCR Assistant.

User Question:
{question}

Retrieved Document Context:
{context if context else "No document context retrieved."}

Your job is to decide whether the user's question is about the uploaded document or not.

Follow these rules strictly.

1. If the user is greeting you or having a normal conversation
Examples:
- hi
- hello
- thanks
- who are you
- what can you do
- tell me a joke
- what is Spring Boot
- what is Python

Ignore the document context completely.

Answer naturally using your own knowledge.

---------------------------------------

2. If the user is asking about the uploaded document

Use ONLY the retrieved context.

Never use outside knowledge.

---------------------------------------

3. If retrieved context exists but is clearly unrelated to the question

Ignore it completely.

Do NOT force document information into your answer.

Answer normally using your own knowledge.

---------------------------------------

4. Only if BOTH are true

• user is asking about the uploaded document

AND

• the answer does not exist in the retrieved context

Then reply naturally like

"I couldn't find that information in the uploaded document."

---------------------------------------

Never mention

- context
- prompt
- retrieval
- vector search
- internal reasoning

Return ONLY the final answer.
"""
        
    # 5. Log lengths and model name (Do not log content for privacy)
    logger.info(f"Model name: {MODEL_NAME}")
    logger.info(f"Question length: {len(question)} characters")
    logger.info(f"Context length: {len(context)} characters")
    
    # 10. Simple retry mechanism
    max_retries = 1
    retry_delay = 2.0
    
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": final_prompt,
        "system": SYSTEM_INSTRUCTION,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_predict": 2048
        }
    }
    
    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            
            elapsed_time = time.time() - start_time
            # 5. Log response time and success
            logger.info(f"Successfully generated answer in {elapsed_time:.3f} seconds (Attempt {attempt + 1}).")
            
            # 4. Safer response extraction
            data = response.json()
            raw_answer = data.get("response", "")
            
            if not raw_answer:
                logger.error("Ollama returned an empty or invalid text response.")
                return "An error occurred while generating the answer."
            
            # 9. Clean the returned answer (normalize line endings and remove whitespace)
            clean_answer = raw_answer.strip().replace("\r\n", "\n")
            
            if not clean_answer:
                logger.error("Cleaned Ollama response was empty.")
                return "An error occurred while generating the answer."
                
            return clean_answer
            
        except requests.RequestException as e:
            # 5. Log failures
            logger.error(f"An error occurred while calling the Ollama API (Attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                logger.error("All attempts to call Ollama API failed.")
                return "An error occurred while generating the answer."
        except Exception as e:
            logger.error(f"An unexpected error occurred (Attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                logger.error("All attempts to call Ollama API failed.")
                return "An error occurred while generating the answer."
                
    return "An error occurred while generating the answer."
