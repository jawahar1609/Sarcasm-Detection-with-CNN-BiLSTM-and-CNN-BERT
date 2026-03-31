# Hybrid Sarcasm Detection - Simplified Pipeline

Clean, streamlined pipeline for hybrid deep learning sarcasm detection with automatic data management.

## Quick Start

### 1. Setup Environment

```bash
cd hybrid-sarcasm-detection
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline - One Command!

```bash
# Run everything for a specific model and dataset
python run.py --model cnn --dataset sarcasm_news

# Or run all models on all datasets
python run.py
```

**That's it!** The pipeline automatically:
- Downloads raw data (if needed)
- Preprocesses data
- Trains the model
- Evaluates performance
- Generates XAI visualizations
- Stores all results

---

## Usage Examples

### Basic Usage

```bash
# Run specific model on specific dataset
python run.py --model cnn --dataset sarcasm_news

# Run with more epochs
python run.py --model cnn_bert --dataset sarc --epochs 15

# Run all models on sarcasm_news dataset
python run.py --dataset sarcasm_news

# Run all models on all datasets
python run.py
```

### Advanced Options

```bash
# Skip training if models already exist
python run.py --skip-train

# Skip XAI analysis (faster)
python run.py --skip-xai

# Custom batch size
python run.py --model lstm --batch-size 128

# Skip evaluation
python run.py --skip-eval
```

---

## Available Models

- `cnn` - CNN baseline
- `lstm` - LSTM baseline  
- `bert` - BERT-based model
- `cnn_bilstm` - CNN + BiLSTM hybrid
- `cnn_bert` - CNN + BERT hybrid (best performance)

## Available Datasets

- `sarcasm_news` - News Headlines Dataset (~6MB, automatic download)
- `sarc` - SARC Dataset (~500MB, automatic download)

---

## Project Structure

```
hybrid-sarcasm-detection/
├── run.py                    # Main unified pipeline orchestrator
├── scripts/                  # All processing scripts
│   ├── train.py             # Training logic
│   ├── evaluate.py          # Evaluation logic
│   └── xai.py               # XAI analysis logic
├── preprocessing/            # Data preprocessing pipeline
│   ├── download.py          # Auto-download datasets
│   ├── data_pipeline.py     # Unified preprocessing
│   └── ...
├── models/                   # Model implementations
│   ├── cnn.py
│   ├── lstm.py
│   ├── bert.py
│   ├── cnn_bilstm.py
│   └── cnn_bert.py
├── raw_data/                 # Downloaded datasets (auto-created)
└── results/                  # All outputs (auto-created)
    ├── models/              # Trained model checkpoints (.pt)
    ├── metrics/             # Evaluation results (JSON)
    └── xai/                 # XAI visualizations (PNG)
```

---

## Pipeline Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Download  │ -> │ Preprocess  │ -> │    Train    │ -> │  Evaluate   │ -> │     XAI     │
│  Raw Data   │    │    Data     │    │   Model     │    │   Model     │    │  Analysis   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      ↓                   ↓                   ↓                   ↓                   ↓
  raw_data/         In-Memory          results/models/    results/metrics/    results/xai/
```

**Key Features:**
- **No fallback logic** - Simple, predictable flow
- **No intermediate files** - Preprocessing happens in-memory
- **Automatic downloads** - No manual data setup needed
- **Organized results** - All outputs sorted by type

---

## Results

After running the pipeline, your results will be organized as:

```
results/
├── models/
│   ├── CNN_sarcasm_news_model.pt
│   ├── HybridCNNBert_sarcasm_news_model.pt
│   └── ...
├── metrics/
│   ├── CNN_sarcasm_news_results.json
│   ├── HybridCNNBert_sarcasm_news_results.json
│   └── ...
└── xai/
    ├── CNN_sarcasm_news/
    │   ├── sample_1_CNN_sarc.png
    │   ├── sample_2_CNN_nonsarc.png
    │   └── ...
    ├── HybridCNNBert_sarcasm_news/
    │   └── ...
    └── ...
```

---

## What Changed?

This is a **simplified, optimized version** of the codebase:

### Before
- Multiple entry points (`train.py`, `evaluate.py`, `xai_analysis.py`)
- Complex fallback logic throughout
- Manual data path management
- Scattered results

### After
- **Single entry point** (`run.py`)
- **Zero fallback logic** - Clean, predictable code
- **Automatic everything** - Download, preprocess, train, evaluate, XAI
- **Organized results** - Everything in its place

---

## Tips

1. **First run:** Start with a single model to test the pipeline
   ```bash
   python run.py --model cnn --dataset sarcasm_news --epochs 5
   ```

2. **Development:** Use `--skip-train` to reuse existing models
   ```bash
   python run.py --skip-train
   ```

3. **Production:** Run all models with more epochs
   ```bash
   python run.py --epochs 20
   ```

4. **Quick testing:** Skip XAI for faster iteration
   ```bash
   python run.py --skip-xai
   ```

---

## Requirements

See `requirements.txt`. Main dependencies:
- PyTorch
- transformers
- scikit-learn
- pandas
- matplotlib
- kagglehub (for automatic downloads)

---

## License

MIT License - see LICENSE file for details