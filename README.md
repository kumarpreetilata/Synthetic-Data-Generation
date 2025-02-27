# Synthetic Text Data Generation using LLMs (Kaggle-Compatible)

## Overview
This project generates **synthetic corporate training scripts** using a **4-bit quantized LLM**. It demonstrates how to:
- Load and run an LLM in **4-bit quantization** to save memory.
- Use **Hugging Face Transformers** for text generation.
- Execute everything in a **Kaggle Notebook** without memory issues.

---

## Setup Instructions

### **Step 1: Install Dependencies**
Run the following in your Kaggle Notebook to install the required libraries:

```python
!pip uninstall -y torch torchvision torchaudio fastai pylibcugraph-cu12 pylibraft-cu12 rmm-cu12 bitsandbytes cuml-cu12 raft-dask-cu12
!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 fastai
!pip install bitsandbytes transformers accelerate --no-cache-dir
!pip install pylibraft-cu12==25.2.0 rmm-cu12==25.2.0 pylibcugraph-cu12 cuml-cu12 raft-dask-cu12 --no-cache-dir
```

### **Step 2: Login to Hugging Face**


## Running the Model

### **Load Model & Generate Text**


## Why Use This Approach?
- **4-bit quantization** → Saves VRAM & runs on Kaggle GPU.
- **Auto Device Mapping** → No manual `device` settings needed.
- **Realistic Training Data** → Custom corporate training simulations.

---

## File Structure
```
synthetic-text-data-generation
│── synthetic-text-data-generation.ipynb  # Main Kaggle Notebook
│── README.md                              # Project Documentation
│── training_script.txt                    # Generated synthetic text output
```

---

## License
This project is licensed under the **MIT License**.


