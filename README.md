# Bloodraven: Your Personal Greenseer

## Introduction

Bloodraven is an AI-powered chatbot that brings the enigmatic character from George R.R. Martin's "A Song of Ice and Fire" (ASOIAF) universe to life. This project uses Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to create an interactive experience for ASOIAF fans.

## Features

- **Character Embodiment**: Interact with Bloodraven, the legendary greenseer and spymaster.
- **ASOIAF Knowledge Base**: Access a vast repository of information from the ASOIAF universe.
- 
## Technical Overview

### RAG Implementation

The chatbot uses RAG to enhance its responses:

1. **Web Scraping**: Data extracted from Wiki of Westeros and Wikipedia using BeautifulSoup and WikipediaAPI.
2. **Text Processing**: Documents split using Langchain's RecursiveCharacterTextSplitter.
3. **Embeddings**: BAAI/bge-base-en-v1.5 model used for generating text embeddings.
4. **Vector Store**: FAISS employed for fast similarity search and retrieval.

### Model Selection

- **Base Model**: zephyr-7b-beta, fine-tuned from mistralai/Mistral-7B-v0.1.
- **Quantization**: Implemented using BitsAndBytes library for efficiency.

### User Interface

Gradio is used to create an intuitive interface for user interactions.

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open the provided URL in your browser
5. Start asking questions to Bloodraven!

## Future Developments

- Comparison between RAG and fine-tuning approaches
- Expansion of the knowledge base
- Enhanced character embodiment through advanced prompt engineering

## Conclusion

Bloodraven is more than just a chatbot; it's a gateway to the rich and mysterious world of ASOIAF. Whether you're seeking answers about Westerosi lore or simply want to experience the cryptic wisdom of a greenseer, Bloodraven is your personal guide to the realms of ice and fire.

Citations:
[1] https://huggingface.co/spaces/Yadukrishnan/Bloodraven




---
title: Bloodraven
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
license: apache-2.0
short_description: Bloodraven answers your questions about the world of ASOIAF!
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).
