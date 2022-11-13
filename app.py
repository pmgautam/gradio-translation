# imports
import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")


def predict(text):
    """_summary_
    predict function to do translation task
    """
    text = [text]
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["npi_Deva"], max_length=30
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


# call gradio interface
examples = ["use this example to see translation in nepali",
            "this text is to test english to nepali translation"]
gr.Interface(fn=predict,
             inputs=gr.Textbox(),
             outputs=gr.Textbox(),
             examples=[examples]).launch()
