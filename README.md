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

```python
!pip install -q transformers huggingface_hub
from huggingface_hub import login

# Paste your Hugging Face Access Token
login("hf_your_access_token_here")
```

---

## Running the Model

### **Load Model & Generate Text**

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Define model name
model_name = "perplexity-ai/r1-1776"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use FP16 for better memory handling
    load_in_4bit=True,          # Enable 4-bit quantization
    device_map="auto",          # Automatically assigns GPU/CPU
    trust_remote_code=True
)

# Create text-generation pipeline (No device argument needed)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define prompt
prompt = """
Generate a corporate training script for customer service employees.
Include a trainer and a trainee discussing how to build interactive data visualization using Tableau/PowerBI.
Provide a realistic conversation format.
"""

# Generate text
generated_text = llm(prompt, max_length=500, num_return_sequences=1)[0]["generated_text"]

# Print output
print("\nGenerated Script:\n", generated_text)

# Save output to file
with open("training_script.txt", "w") as f:
    f.write(generated_text)

print("\nSynthetic training script saved as 'training_script.txt'")
```

---

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

Feel free to contribute and improve the repository.

