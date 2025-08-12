# Global-Refugee-Rights-and-Services-Agent

This project is an AI-powered **Refugee Camp Policy Assistant** designed to help humanitarian workers, government officials, and refugees access up-to-date regulations and guidelines in a conversational format. It uses **Mistral-7B-Instruct** for reasoning and **FAISS** for semantic search over official refugee policy documents.

The current version focuses on **Kenya's Refugees General Regulations (2024)** but is designed to scale to policies from other countries in the future.

---

## **Features**

* **Question Answering:** Ask policy-related questions in plain language.
* **Document Search:** Retrieves relevant sections from the PDF regulations.
* **Optimized for Low-Cost GPUs:** Runs on Google Colab Free Tier (T4 GPU, 12GB VRAM) using 4-bit quantization.
* **Streamlit Interface:** Simple, user-friendly web app that can be shared via ngrok.

---

## **Repository Contents**

| File                                        | Description                                                   |
| ------------------------------------------- | ------------------------------------------------------------- |
| `refugee_policy.ipynb`                      | Jupyter Notebook for development and testing in Google Colab. |
| `app.py`                                    | Streamlit app to deploy the model with a web interface.       |
| `THE-REFUGEES-GENERAL-REGULATIONS-2024.pdf` | Kenya Refugee Policy reference document.                      |

---

## **Quick Start (Google Colab)**

1. **Clone this repository** inside a Colab notebook:

```bash
!git clone https://github.com/YOUR_USERNAME/refugee-policy-assistant.git
%cd refugee-policy-assistant
```

2. **Open `refugee_policy.ipynb`** and follow the steps to:

   * Install dependencies
   * Load the policy document
   * Run the Mistral-7B-Instruct model
   * Ask questions directly in the notebook

3. **Required Accounts:**

   * **Hugging Face** (to load the Mistral model)

     * Sign up: [https://huggingface.co/join](https://huggingface.co/join)
     * Get an access token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
       In the notebook, log in:

       ```python
       from huggingface_hub import login
       login()
       ```
   * **ngrok** (for sharing the Streamlit app)

     * Sign up: [https://ngrok.com](https://ngrok.com)
     * Get your auth token: [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
       Set in Colab:

       ```python
       from pyngrok import ngrok
       ngrok.set_auth_token("YOUR_NGROK_TOKEN")
       ```

---

## **Running the Streamlit App in Colab**

1. Make sure `app.py` and the PDF are in the same folder.
2. Install and start the app:

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

public_url = ngrok.connect(8501)
print(f"Access your app here: {public_url}")

!streamlit run app.py --server.port 8501 &>/dev/null&
```

3. The **public URL** printed will give you instant access to your app.

---

## **Development Notes**

* **Model:** `mistralai/Mistral-7B-Instruct-v0.2` from Hugging Face
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database:** FAISS
* **Chunking:** 1000 characters with 150 overlap for balanced retrieval
* **Temperature:** 0.01 for deterministic outputs

---

## **Planned Next Steps**

* Add refugee policy documents from more countries
* Support multilingual queries and responses
* Improve response summarization for mobile use

---

## **License**

MIT License — feel free to use and adapt with attribution.

---

