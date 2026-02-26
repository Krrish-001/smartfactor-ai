# smartfactor-ai
SmartFactory AI
Production-Grade LLM + Vision Intelligence System for Manufacturing
🔹 Overview

SmartFactory AI is an end-to-end AI system integrating:

QLoRA fine-tuned LLMs (LLaMA 3 / Mistral)

Retrieval-Augmented Generation (FAISS)

Vision Transformer-based defect detection

Automated evaluation pipeline

Dockerized API deployment

Built for industrial manufacturing intelligence applications.

🔹 Features

LLM-based defect report summarization

Engineering document QA (RAG)

PCB defect detection (CNN + ViT)

Hallucination detection pipeline

Grad-CAM explainability

Quantized 4-bit fine-tuning

Fully containerized deployment

🔹 Tech Stack

Python, PyTorch, HuggingFace Transformers, PEFT, FAISS, OpenCV, FastAPI, Docker

🔹 Quick Start
git clone https://github.com/yourname/smartfactory-ai
cd smartfactory-ai
pip install -r requirements.txt
🔹 Run API
uvicorn api.main:app --reload
