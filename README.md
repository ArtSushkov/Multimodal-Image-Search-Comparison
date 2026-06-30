# Image Search by Text Description

Comparative analysis of classical and vision-language model (CLIP) approaches for semantic image retrieval. Demonstrates why joint multimodal training is essential for cross-modal retrieval tasks.

## Overview

This project builds and compares two fundamentally different pipelines for retrieving images by text description:

- **Classical approach**: Independent ResNet-18 (image) + SBERT (text) encoders with concatenated embeddings fed into a neural network regressor
- **CLIP-based approach**: Zero-shot CLIP inference and LoRA fine-tuning on domain-specific image-text pairs

The key finding is that CLIP zero-shot (Recall@1 = 0.832) dramatically outperforms the classical pipeline (Recall@1 = 0.120), demonstrating that joint vision-language pre-training is critical for cross-modal retrieval.

## Key Results

| Approach | Recall@1 | Recall@5 | Trainable Params |
|----------|----------|----------|-----------------|
| NeuralNet (ResNet+SBERT) | 0.120 | 0.368 | ~124K |
| CLIP Zero-Shot | 0.832 | 0.994 | 0 (frozen) |
| CLIP + LoRA FT | 0.836 | 0.982 | ~0.9M |

LoRA fine-tuning (r=16, alpha=32, 10 epochs, 37K pairs) yielded only marginal improvement over zero-shot, indicating that the training dataset volume is the bottleneck rather than the adaptation method.

## What This Project Demonstrates

- **VLM fine-tuning**: Hands-on experience with CLIP model adaptation using PEFT/LoRA
- **InfoNCE contrastive loss**: Manual implementation for CLIP training when using PEFT wrappers
- **Evaluation methodology**: Recall@k ranking metrics, score-distribution analysis, rank histograms
- **Data curation**: Merging expert and crowd-sourced annotations, handling label imbalance, legal content filtering
- **Embedding analysis**: Comparison of independently trained vs jointly trained embedding spaces

## Tech Stack

- **PyTorch** — model training, custom neural network architectures, LoRA fine-tuning
- **Transformers (HuggingFace)** — CLIP model loading and inference
- **PEFT** — LoRA configuration and fine-tuning
- **Sentence Transformers** — SBERT text embeddings
- **TorchVision** — ResNet-18 image feature extraction
- **scikit-learn** — evaluation metrics, data splitting, preprocessing
- **PIL / OpenCV** — image loading and processing

## Project Structure

```
.
├── multimodal_image_search_comparison.ipynb   # Main notebook
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in Google Colab or Jupyter (GPU recommended)
4. Update `PATH_TO_ZIP_FILE` and `DIRECTORY_TO_EXTRACT_TO` with your dataset paths
5. Run all cells sequentially

> **Note**: The dataset is not included in this repository due to licensing.

## Notebook Sections

1. **Exploratory Data Analysis** — train/test split structure, annotation analysis, target variable formation
2. **Legal Compliance** — filtering content with minors per project requirements
3. **Classical Pipeline** — ResNet-18 + SBERT vectorization, model comparison (LinearRegression, DecisionTree, CatBoost, NeuralNet)
4. **CLIP Zero-Shot** — inference without fine-tuning, Recall@k evaluation
5. **CLIP + LoRA Fine-Tuning** — PEFT-based adaptation with contrastive loss
6. **Comparative Analysis** — side-by-side evaluation, rank histograms, score distributions, LoRA impact analysis
