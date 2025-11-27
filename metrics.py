# metrics.py
# ----------------------------------------------------------
# Full performance evaluation for GPT-2 with ROUGE added
# ----------------------------------------------------------

import argparse
import json
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# thop for FLOPs / MACs
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# ROUGE scorer
from rouge_score import rouge_scorer


# ----------------------------------------------------------
# Load JSONL Input (row -> input)
# ----------------------------------------------------------
def load_data(path):
    samples = []
    targets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row = obj.get("row", obj)

            inp = row.get("input")
            tgt = row.get("target")

            if inp:
                samples.append(inp)
                targets.append(tgt)
    return samples, targets


# ----------------------------------------------------------
# Tokenizer wrapper
# ----------------------------------------------------------
def tokenize_batch(tokenizer, text_list, device, max_length=256):
    enc = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


# ----------------------------------------------------------
# CE, PPL, Token Accuracy
# ----------------------------------------------------------
def compute_lm_metrics(model, tokenizer, texts, device, max_length=256, batch_size=8):
    losses = []
    correct = 0
    total = 0

    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenize_batch(tokenizer, batch_texts, device, max_length)

        labels = enc["input_ids"]
        with torch.no_grad():
            out = model(**enc, labels=labels)
            loss = out.loss
            logits = out.logits  # (B, L, V)

        vocab_size = logits.size(-1)
        logits_shifted = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        labels_shifted = labels[:, 1:].contiguous().view(-1)

        if tokenizer.pad_token_id is None:
            mask = torch.ones_like(labels_shifted, dtype=torch.bool)
        else:
            mask = labels_shifted != tokenizer.pad_token_id

        preds = torch.argmax(logits_shifted, dim=-1)
        correct += (preds[mask] == labels_shifted[mask]).sum().item()
        total += mask.sum().item()

        losses.append(loss.item())

    ce = float(np.mean(losses))
    ppl = float(np.exp(ce)) if ce < 50 else float("inf")
    acc = float(correct / total) if total else 0.0
    return ce, ppl, acc


# ----------------------------------------------------------
# Parameter Count
# ----------------------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------
# FLOPs / MACs — using thop (no FloatTensor bug)
# ----------------------------------------------------------
def compute_flops(model, seq_len=128):
    if not HAS_THOP:
        print("⚠️ thop not installed; skipping FLOPs/MACs.")
        return None, None

    device = next(model.parameters()).device
    dummy_input = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, seq_len),
        dtype=torch.long,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    flops = macs * 2
    return float(flops), float(macs)


# ----------------------------------------------------------
# Latency, Throughput, GPU Memory
# ----------------------------------------------------------
def benchmark_inference(model, tokenizer, texts, device, max_length=256, batch_size=8, max_batches=20):
    latencies = []
    tokens_processed = 0

    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batches_run = 0

    for i in range(0, len(texts), batch_size):
        if batches_run >= max_batches:
            break

        batch_texts = texts[i : i + batch_size]
        enc = tokenize_batch(tokenizer, batch_texts, device, max_length)
        tokens_processed += int(enc["input_ids"].numel())

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append(end - start)
        batches_run += 1

    if not latencies:
        return None, None, None, None

    latencies = np.array(latencies)
    median = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    total_time = float(np.sum(latencies))
    throughput = float(tokens_processed / total_time) if total_time > 0 else None

    peak_mem = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    return median, p95, throughput, peak_mem


# ----------------------------------------------------------
# ROUGE computation
# ----------------------------------------------------------
def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1 = []
    r2 = []
    rL = []

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rL)),
    }


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main(args):
    device = torch.device(args.device)

    print("\n📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    print("\n📥 Loading JSONL:", args.input_path)
    texts, refs = load_data(args.input_path)
    print("✔ Loaded samples:", len(texts))

    # LANGUAGE MODEL METRICS
    print("\n📊 Computing CE / PPL / Token Accuracy...")
    ce, ppl, accuracy = compute_lm_metrics(model, tokenizer, texts, device)

    # PARAM COUNT
    print("\n🧮 Counting parameters...")
    total_params, trainable_params = count_parameters(model)

    # FLOPs / MACs
    print("\n⚙ Computing FLOPs/MACs...")
    flops, macs = compute_flops(model, seq_len=128)

    # LATENCY / THROUGHPUT
    print("\n🚀 Benchmarking inference...")
    median, p95, throughput, peak_mem = benchmark_inference(
        model, tokenizer, texts, device
    )

    # ROUGE
    print("\n📝 Generating predictions for ROUGE...")
    preds = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))

    print("\n📊 Computing ROUGE scores...")
    rouge_results = compute_rouge(preds, refs)

    # MODEL SIZE
    model_size_mb = sum(
        os.path.getsize(os.path.join(args.model_dir, f))
        for f in os.listdir(args.model_dir)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ) / (1024 ** 2)

    # FINAL REPORT
    print("\n===============================")
    print("📈 PERFORMANCE REPORT")
    print("===============================")
    print(f"Cross-Entropy:          {ce:.4f}")
    print(f"Perplexity:             {ppl:.4f}")
    print(f"Token Accuracy:         {accuracy:.4f}")

    print("\nText Generation Metrics:")
    print(f"  ROUGE-1 F1:           {rouge_results['rouge1']:.4f}")
    print(f"  ROUGE-2 F1:           {rouge_results['rouge2']:.4f}")
    print(f"  ROUGE-L F1:           {rouge_results['rougeL']:.4f}")

    print("\nModel Parameters:")
    print(f"  Total Params:         {total_params:,}")
    print(f"  Trainable Params:     {trainable_params:,}")

    print("\nInference:")
    print(f"  Median Latency:       {median}")
    print(f"  P95 Latency:          {p95}")
    print(f"  Throughput:           {throughput}")
    print(f"  Peak GPU Memory:      {peak_mem}")

    print("\nCompute:")
    print(f"  MACs:                 {macs}")
    print(f"  FLOPs:                {flops}")

    print("\nModel Size:")
    print(f"  Model size:           {model_size_mb:.2f} MB")
    print("\n✅ DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
