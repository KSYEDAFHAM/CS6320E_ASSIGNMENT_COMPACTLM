# infer_telugu.py
# Works with your custom-trained GPT-2 (with <SEP> and <PAD>)

import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path):
    """Load JSONL with your format: {'row': {'input': ...}}"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # Your dataset structure
            row = obj.get("row", obj)

            text = row.get("input")
            if text is None:
                raise KeyError(f"No 'input' field in: {obj}")

            samples.append(text)

    return samples


def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    print(f"\n📥 Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    # IMPORTANT: match training vocab size
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    print(f"📥 Loading input JSONL: {args.input_path}")
    inputs = load_data(args.input_path)
    print(f"✔ Loaded {len(inputs)} samples")

    outputs = []

    print("\n🔮 Running inference...")

    for batch in batchify(inputs, args.batch_size):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        # Next-token prob from last position
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        for i, p in enumerate(probs):
            outputs.append({
                "input": batch[i],
                "next_token_probs": p.cpu().tolist()
            })

    print("\n💾 Saving results to:", args.output_path)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row) + "\n")

    print("\n🎉 Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir", type=str,
        default=r"C:\Users\sheej\OneDrive\Desktop\MTECH AIDA\SEM1\NLP\finaloutput\gpt2_telugu_refined_2",
        help="Path to trained GPT-2 model"
    )
    parser.add_argument(
        "--input_path", type=str,
        default=r"C:\Users\sheej\OneDrive\Desktop\MTECH AIDA\SEM1\NLP\finaloutput\infer_test_telugu.jsonl",
        help="Path to input JSONL"
    )
    parser.add_argument(
        "--output_path", type=str,
        default=r"C:\Users\sheej\OneDrive\Desktop\MTECH AIDA\SEM1\NLP\finaloutput\infer_telugu.jsonl",
        help="Output JSONL file"
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")  # safer default
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
