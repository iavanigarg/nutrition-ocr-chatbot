# ==========================================================
# OCR SERVICE
#
# Kaam:
# Image --> OvisOCR2 --> Markdown/Text
# ==========================================================

from PIL import Image
import torch
import re

from transformers import (
    AutoProcessor,
    AutoModel
)

# ----------------------------------------------------------
# Hugging Face Model
# ----------------------------------------------------------

MODEL_NAME = "ATH-MaaS/OvisOCR2"

print("\nLoading OvisOCR2 Processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Processor Loaded Successfully!")

print("\nLoading OvisOCR2 Model...")

# CPU use kar rahe hain
# Agar GPU hota to device_map="auto" use karte

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype=torch.float32
)

print("Model Loaded Successfully!")


# ==========================================================
# OCR FUNCTION
# ==========================================================

def extract_text(image_path):
    """
    Input:
        output/page_1.png

    Output:
        OCR Extracted Markdown/Text
    """

    # ------------------------------------------------------
    # Image Load
    # ------------------------------------------------------

    image = Image.open(image_path)

    # ------------------------------------------------------
    # OCR Prompt
    # ------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all readable content from the image in natural human reading order "
                        "and output the result as a single Markdown document. For charts or images, "
                        "represent them using an HTML image tag: <img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, "
                        "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
                        "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
                        "Transcribe all other text as standard Markdown. Preserve the original text "
                        "without translation or paraphrasing."
                    )
                }
            ]
        }
    ]

    # ------------------------------------------------------
    # Convert Image + Prompt into Model Inputs
    # ------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # CPU par inputs bhejo
    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    print("Running OCR...")

    # ------------------------------------------------------
    # OCR Generation
    # ------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # ------------------------------------------------------
    # Decode Output
    # ------------------------------------------------------

    result = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    # ------------------------------------------------------
    # Clean Output
    # ------------------------------------------------------

    # Remove <think>...</think> blocks
    result = re.sub(
        r"<think>.*?</think>",
        "",
        result,
        flags=re.DOTALL
    )

    # Remove extra blank lines
    result = re.sub(r"\n\s*\n+", "\n\n", result)

    # Remove leading/trailing spaces
    result = result.strip()

    return result