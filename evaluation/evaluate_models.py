from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_DIR = "../models/bilingual-lm-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda")

def generate(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_prompts = [
        "বাংলাদেশের রাজধানী হলো",
        "আজকের আবহাওয়া কেমন"
    ]
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        print(f"Generated: {generate(prompt)}\n")
