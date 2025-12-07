# LLM Evaluation UI

Interactive Streamlit dashboard for testing and comparing LLM models on HumanEval benchmark.

## Quick Reference

| Action | Command | URL |
|--------|---------|-----|
| **Start** Streamlit | `./run.sh` | http://localhost:8502 |
| **Start** MLflow | `./run_mlflow.sh` | http://localhost:5000 |
| **Stop** Everything | `./stop.sh` | - |

## Quick Start

### Option 1: Easy Launch (Recommended)

**Start Streamlit App:**
```bash
cd UI/llm-eval-ui
./run.sh
```

**Start MLflow UI (optional, in separate terminal):**
```bash
cd UI/llm-eval-ui
./run_mlflow.sh
```

### Option 2: Manual Launch

**Streamlit App:**
```bash
cd UI/llm-eval-ui
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/streamlit run app.py
```

**MLflow UI (optional, separate terminal):**
```bash
cd /mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/mlflow ui --port 5000
```

### URLs

- **Streamlit App:** http://localhost:8502
- **MLflow UI:** http://localhost:5000 (if started)

## Stopping the Application

### Quick Stop (Recommended)

```bash
cd UI/llm-eval-ui
./stop.sh
```

This will:
- Stop Streamlit app
- Stop MLflow UI
- Free ports 8501 and 5000
- Verify everything is stopped cleanly

### Manual Stop

**In each terminal running the apps:**
```bash
Ctrl+C
```

**If processes don't stop:**
```bash
# Kill Streamlit
pkill -f "streamlit run app.py"

# Kill MLflow
pkill -f "mlflow ui"

# Free ports if still in use
fuser -k 8501/tcp  # Streamlit port
fuser -k 5000/tcp  # MLflow port
```

**Verify shutdown:**
```bash
# Check no processes running
ps aux | grep streamlit
ps aux | grep mlflow

# Check ports are free
lsof -i :8501
lsof -i :5000
```

## Features

### 🧪 Interactive Testing
- Select from 3 models: Student (8B), Distilled (8B+LoRA), Teacher (70B)
- Choose any HumanEval problem (164 available)
- Adjust generation parameters (temperature, max tokens)
- Generate multiple solutions
- Auto-log to MLflow

### 📊 Model Comparison
- Compare outputs side-by-side (coming soon)
- Run test cases
- Show pass/fail results

### 📈 Past Results
- View evaluation metrics from `results/` directory
- See pass@k performance
- Check energy consumption
- Review full JSON results

## Project Structure

```
llm-eval-ui/
├── app.py                      # Main Streamlit application
├── eval_helper.py              # HumanEval evaluation wrapper
├── config.py                   # Configuration (paths, models)
├── requirements.txt            # Python dependencies
├── run.sh                      # Start Streamlit app
├── run_mlflow.sh              # Start MLflow UI
├── stop.sh                     # Stop all processes (NEW)
├── README.md                   # This file
├── README_FIX.md              # Detailed bug fix documentation
└── test_*.py                   # Test scripts
```

## Configuration

Edit `config.py` to change:
- Model paths
- MLflow tracking URI
- Default generation parameters
- Results directory

## Models Available

1. **Student (8B Base)**
   - Llama-3.1-8B-Instruct (no fine-tuning)
   - Fast, baseline performance

2. **Distilled (8B + LoRA)**
   - Llama-3.1-8B with trained LoRA adapters
   - Knowledge distilled from 70B teacher
   - Same speed as student, better performance

3. **Teacher (70B - 8bit)**
   - Llama-3.1-70B-Instruct (8-bit quantized)
   - Best performance, slower, higher cost

## Troubleshooting

### Models not found
Models are automatically loaded from `~/.cache/huggingface/hub/`. If not present, they'll be downloaded on first use.

### LoRA adapters not found
Check that the path in `config.py` points to:
`outputs/llama31_8b_kd_lora/lora/`

### HumanEval not found
Install with: `pip install human-eval`

### MLflow not working
Make sure `mlruns/` directory exists in the project root.

## Tips

- Use temperature 0.01 for deterministic code generation
- Start with 1 sample for quick testing
- Generate 10 samples to calculate pass@k metrics
- Enable MLflow logging to track all experiments


=====================
# other inportant commands

ps aux | grep ptuemler
ps aux | grep ptuemler | grep python

pkill -u ptuemler python

or 

sudo pkill -u [username] : If you got super user privileges
sudo pkill -9 -u [username]


kill [PID]	
kill -u [PID]