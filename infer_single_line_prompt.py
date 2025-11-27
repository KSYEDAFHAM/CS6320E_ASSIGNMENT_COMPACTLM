# inference_telugu_gpt2.py
# -------------------------------------
# Load fine-tuned GPT-2 Telugu model and generate text
# -------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    print("⏳ Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print("✅ Model loaded:", model_path)
    return tokenizer, model


def generate_text(tokenizer, model, prompt, max_new_tokens=60, temperature=0.9, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    # -----------------------------
    # Set your saved model folder
    # -----------------------------
    model_path = r"C:\Users\sheej\OneDrive\Desktop\MTECH AIDA\SEM1\NLP\finaloutput\gpt2_telugu_refined_2"

    tokenizer, model = load_model(model_path)

    # -----------------------------
    # Example Inputs
    # -----------------------------
    prompt = "ప్రకటించింది.  పేద మరియు వెనుకబడిన వ"   # Telugu + <SEP> format

    output = generate_text(tokenizer, model, prompt)

    print("\n🔮 **Generated Text**:\n")
    print(output)
