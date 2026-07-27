# Financial News Sentiment Classification using QLoRA Fine-Tuning

## Overview

This project is an end-to-end Large Language Model (LLM) fine-tuning pipeline for **financial news sentiment classification**. It enables efficient fine-tuning of open-source language models using **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** while minimizing GPU memory requirements. The project also includes a complete inference pipeline, REST API, and Gradio-based web interface for interactive predictions.

The system is designed with a modular architecture, separating configuration, model loading, dataset preprocessing, training, evaluation, inference, backend services, and frontend components. This makes it easy to extend the project to new datasets or base models.

---

## Features

* Fine-tune open-source LLMs using **LoRA** or **QLoRA**
* Support for multiple base models through a configurable model registry
* Automatic loading of previously fine-tuned LoRA adapters
* Resume training from existing adapters
* Dataset preprocessing using Hugging Face Datasets
* Chat-style prompt formatting using tokenizer chat templates
* Automatic model evaluation after every training epoch
* Classification metrics including:

  * Validation Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Training metric visualization
* FastAPI backend for serving predictions
* Gradio frontend for interactive inference
* Configuration through environment variables

---

## Project Architecture

```text
                    ┌───────────────────────┐
                    │      config.py        │
                    └──────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
         model_loader.py            dataset_loader.py
                 │                           │
                 │                           ▼
                 │                    Prompt Formatting
                 │                           │
                 ▼                           ▼
              train.py  ◄────────────── Evaluation
                 │
                 ▼
           LoRA Adapter
                 │
                 ▼
           inference.py
                 │
                 ▼
             FastAPI API
                 │
                 ▼
           Gradio Frontend
```

---

## Workflow

### 1. Dataset Preparation

The financial news dataset is loaded using the Hugging Face `datasets` library.

Each sample is transformed into a chat-style instruction format:

**User**

```
Classify the sentiment of the following financial news sentence as
positive, negative, or neutral.

Sentence:
Stocks rally after strong quarterly earnings.
```

**Assistant**

```
positive
```

This formatting allows instruction-tuned language models to learn the sentiment classification task effectively.

---

### 2. Model Loading

The project supports multiple base models through a centralized configuration.

Examples include:

* Qwen
* Gemma
* TinyLlama

During loading, the system automatically checks whether a previously fine-tuned LoRA adapter exists.

If an adapter is found:

```
Base Model
      +
LoRA Adapter
```

is loaded automatically.

Otherwise, the original base model is loaded.

---

### 3. Fine-Tuning

Training is performed using the Hugging Face TRL `SFTTrainer`.

Depending on the configuration, the project supports:

* Standard LoRA
* QLoRA (4-bit quantized fine-tuning)

Only the lightweight adapter parameters are trained, while the original model weights remain frozen. This significantly reduces GPU memory usage and enables fine-tuning on consumer-grade hardware.

During training, the pipeline automatically:

* trains for the configured number of epochs
* evaluates the model after every epoch
* records training loss
* records validation accuracy
* saves the final LoRA adapter
* stores the tokenizer
* generates a training curve

---

### 4. Evaluation

The evaluation pipeline performs inference on the test dataset and computes standard classification metrics.

Outputs include:

* Validation Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

The confusion matrix is automatically saved for further analysis.

---

### 5. Inference

The inference module loads the appropriate model and generates sentiment predictions for unseen financial news.

If a fine-tuned adapter is available, the project automatically uses the fine-tuned model. Otherwise, inference falls back to the original base model.

---

### 6. Deployment

The project includes two deployment components:

**FastAPI Backend**

Provides REST endpoints for model inference.

* `GET /` – Health check
* `POST /generate` – Generate sentiment prediction

**Gradio Frontend**

A lightweight web interface that communicates with the FastAPI backend, allowing users to classify financial news through a simple browser-based UI.

---

## Project Structure

```text
Financial_News_Classification_LLM/
│
├── backend/
│   ├── api.py
│   ├── dataset_loader.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── model_loader.py
│   └── train.py
│
├── frontend/
│   └── gradio_app.py
│
├── configs/
│   └── config.py
│
├── models/
│
├── .env
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT (LoRA / QLoRA)
* TRL (Supervised Fine-Tuning)
* BitsAndBytes
* FastAPI
* Gradio
* Scikit-learn
* Matplotlib
* Seaborn

---

## Key Capabilities

* Memory-efficient fine-tuning using LoRA and QLoRA
* Modular and reusable project architecture
* Automatic adapter detection and loading
* End-to-end training, evaluation, and inference pipeline
* Interactive web interface for real-time predictions
* Easy switching between supported base models
* Extensible framework for other text classification tasks beyond financial sentiment analysis

---

## Future Improvements

Potential enhancements include:

* Support for additional language models
* Batch inference API
* Model checkpointing after every epoch
* Hyperparameter configuration through YAML files
* Experiment tracking with TensorBoard or Weights & Biases
* Docker containerization
* Automated unit and integration tests
* Deployment using cloud inference services
* Multi-label and multilingual sentiment classification support


1. Training the model
python -m backend.train

2. Evaluating the model
python -m backend.evaluation

3. Start the FastAPI backend
Run:
uvicorn backend.api:app --reload

4. Launch the Gradio frontend
Open a second terminal, navigate to the project root, and run:
python frontend/gradio_app.py
