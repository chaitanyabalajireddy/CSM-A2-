# Leveraging Image-Text Query Fusion for Effective Multi-modal Image Retrieval

## Overview

This project focuses on developing and evaluating OpenCLIP and BLIP-based models for enhanced multi-modal image retrieval, particularly using the FashionIQ dataset. The work emphasizes aligning image and text embeddings for tasks like image-caption matching and retrieval.

### Institution

Ecole Centrale School of Engineering, Mahindra University, Hyderabad (December 2024)

## Contents

1. [Introduction](#introduction)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Architectures](#model-architectures)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Results](#results)
6. [Conclusion](#conclusion)

---

## Introduction

The objective of this project is to enhance image-text retrieval by leveraging the pretrained OpenCLIP and BLIP models. Using the FashionIQ dataset, the models are fine-tuned and evaluated to improve Recall\@10 and Recall\@50 metrics, offering a comparative analysis with state-of-the-art models.

## Dataset Preparation

### FashionIQ Dataset

The dataset consists of:

- **Images**: Fashion-related, categorized (e.g., dresses, tops).
- **Captions**: Human-written descriptions of images.
- **Splits**: Training, validation, and test sets.

### Preprocessing Steps

1. **Dynamic File Matching**: Ensures robustness in locating image files.
2. **Filtering**: Maintains consistency between captions and image subsets.
3. **Transformations**:
   - Resize to 224x224.
   - Convert to tensors.
   - Normalize using ImageNet statistics.
4. **Handling Missing Files**: Includes logging mechanisms to identify missing images.

## Model Architectures

### OpenCLIP

- **Features**:
  - Contrastive learning.
  - Pretrained Vit-B-32 and RN50 architectures.
  - GPU-accelerated processing.
- **Loss Function**:
  - Contrastive loss with hard negative mining.
  - Cosine similarity-based similarity matrix.

### BLIP

- **Features**:
  - Transformer-based vision-language architecture.
  - Enhanced feature alignment for detailed textual descriptions.
- **Preprocessing**:
  - Image and text transformations tailored to BLIP’s requirements.
  - Tokenization for caption alignment.

## Evaluation Metrics

1. **Recall\@k**: Measures the proportion of relevant items retrieved within the top-k predictions.
2. **Precision\@k**: Evaluates the relevance of retrieved results among the top-k predictions.

### Calculation

- **Similarity Matrix**: Dot product of image and text embeddings.
- **Metrics**: Recall\@10, Recall\@50, Precision\@10, and Precision\@50 computed for each category.

## Results

### Quantitative Results (OpenCLIP)

| Model Type | R\@10 | R\@50 | P\@10 | P\@50 |
| ---------- | ----- | ----- | ----- | ----- |
| RN50       | 19.48 | 43.80 | 25.23 | 39.57 |
| Vit-B-32   | 21.17 | 45.62 | 27.65 | 41.02 |

### Quantitative Results (BLIP)

| Model Type | R\@10 | R\@50 | P\@10 | P\@50 |
| ---------- | ----- | ----- | ----- | ----- |
| BLIP       | 25.32 | 50.13 | 32.45 | 46.90 |

### Split-wise Results (OpenCLIP and BLIP)

| Category | Split      | R\@10 | R\@50 | P\@10 | P\@50 |
| -------- | ---------- | ----- | ----- | ----- | ----- |
| Dress    | Validation | 22.5  | 42.1  | 0.45  | 0.35  |
| TopTee   | Validation | 24.2  | 48.8  | 0.50  | 0.40  |
| Shirt    | Validation | 26.1  | 44.5  | 0.40  | 0.30  |

### Comparative Results

| Method               | Dress (R\@10) | Dress (R\@50) | TopTee (R\@10) | TopTee (R\@50) | Shirt (R\@10) | Shirt (R\@50) | Average R\@10 | Average R\@50 |
|----------------------|----------------|----------------|----------------|----------------|---------------|---------------|---------------|---------------|
| TIRG (ViT-B-32)     | 12.85          | 19.05          |                |                |               |               |               |               |
| ComposeAE w/ BERT   | 14.03          | 35.01          |                |                |               |               |               |               |
| SAC w/ BERT         | 26.52          | 51.01          |                |                |               |               |               |               |
| MAAF w/ BERT        | 23.8           | 48.6           |                |                |               |               |               |               |
| CIRPLANT w/ OSCAR   | 17.45          | 40.41          |                |                |               |               |               |               |
| Ours (ViT-B-32)     | 20.67          | 46.59          | 18.55          | 39.04          | 21            | 47.2          | 20.56         | 46.03         |
| Ours (R50)          | 22.35          | 48.89          | 20.3           | 42.1           | 23.5          | 50            | 22.05         | 47.2          |
| Ours (BLIP)         | 25.32          | 50.13          | 37.22          | 32.45          | 46.9          | 39.4          | 25.32         | 50.13         |

## Conclusion

This project demonstrated the effectiveness of OpenCLIP and BLIP models in image-text retrieval tasks using the FashionIQ dataset. The BLIP model, in particular, showed superior performance, indicating the value of joint vision-language embeddings. Future work will explore:

- Integration of multi-modal training.
- Testing on diverse datasets.
- Enhancing hard negative mining techniques.

---

## References

- [FashionIQ Dataset](https://github.com/liuziwei7/fashion-iq)
- [OpenCLIP Documentation](https://github.com/mlfoundations/open_clip)
- [BLIP Documentation](https://huggingface.co/docs/transformers/model_doc/blip)

