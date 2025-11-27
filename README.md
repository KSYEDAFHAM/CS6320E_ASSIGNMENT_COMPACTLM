# Compact Language Model Challenge - Telugu & Marathi

**Course:** CS6320E - Topics in Natural Language Processing  
**Task:** Next-token prediction (Causal Language Modeling)  
**Languages:** Telugu, Marathi

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Team Information](#team-information)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Evaluation Checklist](#evaluation-checklist)

---

## 🎯 Project Overview

This project implements compact, sample-efficient language models for Telugu and Marathi using fine-tuned GPT-2 architecture. The models perform next-token prediction on low-resource Indian languages.

**Key Features:**
- GPT-2 based causal language model
- Early stopping mechanism for optimal performance
- Separate models for Telugu and Marathi
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)

---

## 👥 Team Information

**Members:**
- ASHNA HUSSAIN - M250279CS
- SYED AFHAM K - M250540CS

---

## 🏗️ Model Architecture

### Base Model
- **Architecture:** GPT-2 (Transformer-based decoder)
- **Model Type:** Autoregressive Causal Language Model
- **Base Parameters:** ~124M (GPT-2 base)
- **Tokenizer:** GPT-2 tokenizer with custom tokens (<SEP>, <PAD>)

### Model Configuration
- **Hidden Size:** 768
- **Number of Layers:** 12
- **Attention Heads:** 12
- **Vocabulary Size:** Extended with Telugu/Marathi tokens
- **Maximum Sequence Length:** 256 tokens
- **Total Trainable Parameters:** ~124M + embeddings

### Custom Modifications
- Added special tokens: `<SEP>` (separator), `<PAD>` (padding)
- Resized token embeddings to accommodate new tokens
- Configured pad_token_id for proper masking

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 10GB+ disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/KSYEDAFHAM/CS6320E_ASSIGNMENT_COMPACTLM.git
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Libraries
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
scikit-learn>=1.2.0
pandas>=2.0.0
tqdm>=4.65.0
evaluate>=0.4.0
rouge-score>=0.1.2
bert-score>=0.3.13
nltk>=3.8.0
```

---

## 📊 Dataset

### Dataset Structure
The dataset is provided in JSONL format with the following fields:
- **input:** Source text in Telugu/Marathi
- **target:** Target text for next-token prediction

### Example:
```json
{
  "row": {
    "input": "ఫిబ్రవరి 23, 2019 174 ఎన్నో భారీ అంచనాలతో వచ్చిన మహానుభావుడు ఎన్ . టి . ఆర్ బయోపిక్ రండు పార్టులు నిరాశపరచాయి .",
    "target": "మహానాయకుడు మంచేశాడుగా"
  }
}
```

### Dataset Location
Download from: [Google Drive Link](https://drive.google.com/drive/folders/1d2-Uf2yuCoJ2QLDd03pC-WyBR55JSE0-)

**Expected files:**
- `train_te.jsonl` - Telugu training data
- `validation_te.jsonl` - Telugu validation data
- `train_mr.jsonl` - Marathi training data
- `validation_mr.jsonl` - Marathi validation data

### Preprocessing
The training script automatically:
1. Cleans whitespace and normalizes text
2. Merges input and target with `<SEP>` separator
3. Tokenizes and truncates to max_length (256 tokens)
4. Creates attention masks and labels

---

## 🚀 Training

### Training Script
Use `train_transformer_telugu_gpt2.py` for training.

### Training Telugu Model
```bash
python train_transformer_telugu_gpt2.py \
  --train_path ./data/train_te.jsonl \
  --valid_path ./data/validation_te.jsonl \
  --output_dir ./models/gpt2_telugu \
  --epochs 70 and 30 \
  --batch_size 32 \
  --lr 5e-5 \
  --patience 3
```

### Training Marathi Model
```bash
python train_transformer_telugu_gpt2.py \
  --train_path ./data/train_mr.jsonl \
  --valid_path ./data/validation_mr.jsonl \
  --output_dir ./models/gpt2_marathi \
  --epochs 70 and 15 \
  --batch_size 32 \
  --lr 5e-5 \
  --patience 3
```

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 5e-5 | Initial learning rate |
| Batch Size | 32 | Per-device batch size |
| Gradient Accumulation | 2 | Effective batch size: 64 |
| Epochs | 15 and 75 | Maximum training epochs |
| Warmup Ratio | 0.1 | Learning rate warmup |
| Early Stopping Patience | 3 | Stop after 3 epochs without improvement |
| Max Sequence Length | 256 | Maximum tokens per sequence |
| Optimizer | AdamW | Default from Trainer |
| LR Scheduler | Cosine with warmup | Cosine annealing |
| Max Grad Norm | 1.0 | Gradient clipping |
| FP16 | True | Mixed precision training |
| Gradient Checkpointing | True | Memory optimization |

### Training Strategy
1. **Warmup Phase:** 10% of total steps with linear warmup
2. **Training Phase:** Cosine learning rate decay
3. **Early Stopping:** Monitor validation loss, patience=3
4. **Checkpointing:** Save best model based on eval_loss
5. **Evaluation:** Every epoch

### Hardware Used
- **GPU:** NVIDIA A100 80GB
- **Training Time:** [~X hours for Telugu, ~Y hours for Marathi]
- **Memory Usage:** [Peak GPU memory]

### Random Seeds
- **Training seed:** Set via `PYTHONHASHSEED`, `torch.manual_seed()`, etc.
- **Reproducibility:** Use same seed for consistent results

---

## 🔮 Inference

### Running Inference
```bash
python infer.py \
  --model_path ./models/gpt2_telugu \
  --test_file ./data/test_te.jsonl \
  --output_file ./results/predictions_te.jsonl \
  --batch_size 32 \
  --device cuda:0
```

### Inference Script Usage
```python
# Example usage
from infer import load_model, predict_next_token

# Load model
model, tokenizer = load_model("./models/gpt2_telugu")

# Predict
input_text = "ఫిబ్రవరి 23, 2019"
probabilities = predict_next_token(model, tokenizer, input_text)
```

### Expected Output Format
- Next-token probability distribution
- Top-k predictions with probabilities
- Perplexity score per sample

---

## 📈 Performance Metrics

### Primary Metric
- **Perplexity:** Lower is better (exponential of cross-entropy loss)

### Required Metrics

#### 1. Model Quality Metrics
```bash
python measure_performance.py --model_path ./models/gpt2_telugu --test_file ./data/test_te.jsonl
```

**Computed Metrics:**
- ✅ Perplexity
- ✅ Cross-entropy / Bits-per-token
- ✅ Token-level top-1 accuracy
- ✅ ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- ✅ BLEU Score
- ✅ BERTScore

#### 2. Efficiency Metrics
- **Parameter Count:** Total and trainable parameters
- **Inference Latency:** Median and P95 time per token (ms)
- **Throughput:** Tokens per second
- **FLOPs/MACs:** Estimated operations
- **Model Size:** File size (MB) and quantized size
- **Memory Usage:** Peak GPU/CPU memory during inference

### Metric Calculation Methods
- **Perplexity:** `exp(cross_entropy_loss)`
- **Parameters:** `sum(p.numel() for p in model.parameters())`
- **Latency:** Measured using `time.perf_counter()` over 100 runs
- **Throughput:** `num_tokens / total_time`
- **FLOPs:** Estimated using PyTorch profiler or `thop` library
- **Memory:** Tracked using `torch.cuda.max_memory_allocated()`

---

## 📊 Results

### Telugu Model Performance

| Metric | Value |
|--------|-------|
| Test Perplexity | 1.6068 |
| Cross-Entropy Loss | 0.4742 |

### Marathi Model Performance

| Metric | Value |
|--------|-------|
| Test Perplexity | 2.0059 |
| Cross-Entropy Loss | 0.7169 |

### Sample Outputs

#### Telugu Example
```
Input: "ఫిబ్రవరి 23, 2019 174 ఎన్నో భారీ అంచనాలతో వచ్చిన"
Predicted: "మహానుభావుడు"
Confidence: 0.85
```

#### Marathi Example
```
Input: "दिलेल्या मराठी बातमीवरून मुख्य घटना दर्शवणारी थोडक्यात आणि नेमकी हेडलाईन तयार करा.बातमी:"
Predicted: "दिलेल्या मराठी बातमीवरून मुख्य घटना दर्शवणारी थोडक्यात आणि नेमकी हेडलाईन तयार करा.बातमी: वार्ताहर दिलेल्या मराठी बातमीवरून पो�"
Confidence: 0.85
```

---

## ✅ Evaluation Checklist

### Before Evaluation Session

#### ✅ Technical Preparation
- [ ] Laptop with fully charged battery and charger
- [ ] Complete project code ready to run
- [ ] All environments and dependencies installed locally
- [ ] Models saved separately for Telugu and Marathi
- [ ] Internet independent - all resources available locally
- [ ] Test inference pipeline end-to-end

#### ✅ Model Files
- [ ] Telugu model checkpoint saved
- [ ] Marathi model checkpoint saved
- [ ] Tokenizer files for both languages
- [ ] Configuration files (config.json)

#### ✅ Scripts Ready
- [ ] `infer.py` - Real-time inference on test sets
- [ ] `measure_performance.py` - Compute all metrics
- [ ] Scripts accept command-line arguments
- [ ] File paths easily adjustable

#### ✅ Metrics Scripts
- [ ] Perplexity calculation
- [ ] ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- [ ] BLEU Score
- [ ] METEOR (if applicable)
- [ ] BERTScore (if applicable)
- [ ] Clear display of numerical values
- [ ] Documentation of preprocessing steps

#### ✅ Documentation
- [ ] README.md with clear instructions
- [ ] Report.pdf (4-6 pages)
- [ ] requirements.txt or environment.yml
- [ ] submission_metadata.json
- [ ] Training logs and hyperparameters recorded

### During Evaluation

#### Test Set Evaluation
- [ ] Load provided unseen test set(s)
- [ ] Run model in real-time
- [ ] Generate outputs without code modification
- [ ] Display metrics clearly

### Viva Preparation Topics

#### 1. Model Architecture
- [ ] Explain GPT-2 architecture choice
- [ ] Input/output format justification
- [ ] Tokenization approach (BPE)
- [ ] Embedding technique
- [ ] Hidden layer size: 768
- [ ] Number of layers: 12
- [ ] Total trainable parameters: ~124M

#### 2. Data Handling
- [ ] Dataset sources explained
- [ ] Preprocessing steps documented
- [ ] Text cleaning procedures
- [ ] Data augmentation (if any)
- [ ] Normalization techniques

#### 3. Experimental Setup
- [ ] Training strategy explained
- [ ] Hyperparameters justified:
  - Learning rate: 5e-5
  - Batch size: 32 (effective 64)
  - Dropout: default
  - Optimizer: AdamW
- [ ] Number of epochs: 15 and 70 (with early stopping)
- [ ] Early stopping implementation (patience=3)
- [ ] Checkpointing strategy

#### 4. Model Output Analysis
- [ ] Sample outputs prepared
- [ ] Error analysis ready
- [ ] Strengths identified
- [ ] Weaknesses understood
- [ ] Improvement suggestions

---

## 📁 Project Structure
```
project/
├── models/
│   ├── gpt2_telugu/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── tokenizer files
│   └── gpt2_marathi/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer files
├── data/
│   ├── train_te.jsonl
│   ├── validation_te.jsonl
│   ├── train_mr.jsonl
│   └── validation_mr.jsonl
├── results/
│   ├── predictions_te.jsonl
│   └── predictions_mr.jsonl
├── train_transformer_telugu_gpt2.py
├── infer.py
├── measure_performance.py
├── README.md
├── report.pdf
├── requirements.txt
├── submission_metadata.json
└── environment.yml
```

---

## 🔐 External Resources Declaration

All external resources used in this project:

### Pretrained Models
- **GPT-2 base:** Hugging Face (`gpt2`)
- **License:** MIT License
- **Source:** https://huggingface.co/gpt2

### Libraries & Tools
- **PyTorch:** BSD License
- **Transformers:** Apache 2.0
- **Datasets:** Apache 2.0
- **Standard libraries:** NumPy, Pandas (BSD licenses)

### Datasets
- **Training data:** Provided by course instructors
- **No additional labeled data used**

---

## 📄 Submission Metadata
```json
{
  "team_name": "[Your Team Name]",
  "members": [
    {
      "name": "Ashna Hussain",
      "roll_number": "M250279CS"
    },
    {
      "name": "Syed Afham K",
      "roll_number": "M250540CS"
    }
  ],
  "external_resources": {
    "pretrained_models": ["gpt2"],
    "libraries": ["transformers", "torch", "datasets"],
    "hardware": "[GPU type]"
  },
  "random_seeds": {
    "training": 42,
    "evaluation": 42
  },
  "date_submitted": "2025-11-27"
}
```

---

## 🤝 Collaboration & Academic Integrity

- All code is original to our team
- High-level discussions with other students documented
- No code copying from other teams
- All external resources properly attributed
- No test data leakage into training/validation

---

## 📝 License & Ethical Considerations

### Model License
This model is released for educational purposes only.

### Limitations
- Trained on limited data - may not generalize well
- Potential biases from training data
- Not suitable for production use without further testing
- Language-specific limitations for Telugu and Marathi

### Potential Biases
- Dataset bias toward specific domains
- Underrepresentation of certain dialects
- Historical biases in source text

---

## 📞 Contact

For questions or issues:
- **Email:** ashna_m250279cs@nitc.ac.in or syed_m250279cs@nitc.ac.in
- **Course:** CS6320E - NLP

---

## 🙏 Acknowledgments

- Course instructors and TAs
- Hugging Face for transformers library
- OpenAI for GPT-2 architecture
- Dataset providers

---

**Last Updated:** 27/11/2025  

