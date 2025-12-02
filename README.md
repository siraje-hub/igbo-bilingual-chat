
# 🤖 Igbo-Phi3-Bilingual-Chat (GGUF & Colab Training)

This repository contains the source code, training notebooks, and inference scripts for **Igbo-Phi3-Bilingual-Chat**.

Unlike my previous "Translator" project, this model is a **fully conversational AI**. It can chat, reason, and define words in both Igbo and English, while retaining the general intelligence of the base Phi-3 model.

* **Hugging Face (GGUF):** [nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF](https://huggingface.co/nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF)
* **Base Model:** Microsoft Phi-3-mini-4k-instruct
* **Dataset:** 700k+ mixed examples (Igbo sentences, Dictionary, and UltraChat)

---

## 🚀 How to Run (Python / Local)

You can run this model on your laptop (Windows/Mac/Linux) using Python. It uses the GGUF format, so it is fast and requires very little RAM.

### 1. Install Dependencies
```bash
pip install llama-cpp-python huggingface_hub
````

### 2\. Run the Chat Script

Copy `chat.py` (code provided in this repo) and run it:

```bash
python chat.py
```

-----

## ⚡ Quick Colab Demo

If you don't have a Python environment set up, you can copy-paste this code into a [Google Colab](https://colab.research.google.com/) cell to test the model immediately.

```python
# --- 1. Install Libraries ---
!pip install llama-cpp-python huggingface_hub

# --- 2. Download & Load Model ---
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO_ID = "nwokikeonyeka/Igbo-Phi3-Bilingual-Chat-v1-merged-Q5_K_M-GGUF"
FILENAME = "igbo-phi3-bilingual-chat-v1-merged-q5_k_m.gguf"

print(f"Downloading {FILENAME}...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

# --- 3. Chat Loop ---
print("\n🤖 IGBO CHATBOT READY (Type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']: break
    
    # Correct Phi-3 Prompt Template
    prompt = f"<s><|user|>\n{user_input}<|end|>\n<|assistant|>\n"
    
    output = llm(prompt, max_tokens=256, stop=["<|end|>"], echo=False)
    print(f"AI: {output['choices'][0]['text']}")
```

-----

## 🧠 Training Methodology: "The Colab Relay Race"

Training a full LLM on a free Google Colab GPU usually causes timeouts (disconnects) before the training is finished. To solve this, I used a **"Relay Race"** strategy.

### The Strategy

1.  **Auto-Save:** The training script saves a "Checkpoint" to Hugging Face every 500 steps.
2.  **Resume:** When Colab times out (after \~4 hours), a new session is started.
3.  **Relay:** The script automatically downloads the last checkpoint from Hugging Face and continues training exactly where it left off.

This allowed me to train for **44,500 steps** (over 700k examples) across multiple free sessions without losing progress.

### Datasets Used

To create a "Bilingual Brain" rather than just a translator, I combined three datasets:

1.  **Fluency:** \~522k English-Igbo sentence pairs.
2.  **Vocabulary:** \~5k Igbo Dictionary definitions.
3.  **Memory:** \~200k General Chat conversations (UltraChat) to prevent the model from forgetting English or general logic.

-----

## 🛠️ Prompt Format

This model uses the standard **Phi-3** format. If you use it in other apps (like Ollama or LM Studio), ensure you use this template:

```text
<s><|user|>
{Your Question Here}<|end|>
<|assistant|>
{AI Response Here}<|end|>
```

````
