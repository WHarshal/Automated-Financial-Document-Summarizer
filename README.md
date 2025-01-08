# Automated Financial Document Summarizer

This repository contains a project that utilizes a Transformer-based T5 model to summarize financial documents. The model is fine-tuned on the BillSum dataset, which consists of legislative texts and their summaries. The goal of the project is to generate concise and accurate summaries, making financial and legislative texts more accessible and easier to comprehend.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model and Techniques](#model-and-techniques)
4. [Preprocessing](#preprocessing)
5. [Training](#training)
6. [Evaluation](#evaluation)

---

## Overview
- **Objective**: To build a summarization model for financial and legislative documents using the T5 Transformer architecture.
- **Key Achievements**:
  - Fine-tuned T5-small model on the BillSum dataset.
  - Achieved validation Rouge-1 score of 0.24, Rouge-2 score of 0.10, and Rouge-L score of 0.17.
  - Significantly reduced training loss over three epochs, showcasing effective model fine-tuning.

---

## Dataset

The project uses the **BillSum dataset**, which contains legislative texts from:
- U.S. Congressional bills.
- California state bills.

### Dataset Details
- Each record contains:
  - **bill_id**: Unique identifier for each bill.
  - **text**: Full text of the bill.
  - **clean_text**: Preprocessed version of the text (whitespace removed, basic cleaning applied).
  - **summary**: Short summary of the bill.
  - **title**: Title of the bill.
- Dataset is split into:
  - **us_train_data_final_official.jsonl** (training data).
  - **us_test_data_final_official.jsonl** (testing data for U.S. bills).
  - **ca_test_data_final_official.jsonl** (testing data for California bills).

---

## Model and Techniques

### Model Architecture
- **Model Used**: [T5-small](https://huggingface.co/t5-small)
  - A Transformer-based model designed for text-to-text tasks, making it suitable for summarization.

### Key Features
- **Gradient Checkpointing**: Enabled to reduce memory usage during training.
- **Mixed Precision Training**: Used FP16 for faster training and lower memory requirements.

---

## Preprocessing

### Steps Performed
1. **Text Preparation**:
   - Input: Combined the `clean_text` field with a prefix `"summarize: "` to prepare the data for the T5 model.
   - Target: Used the `summary` field as the expected output.

2. **Tokenization**:
   - Used the T5 tokenizer to encode input and target texts.
   - Applied truncation and padding with max lengths:
     - Input text: 512 tokens.
     - Target text: 150 tokens.

3. **Column Removal**:
   - Dropped unnecessary fields such as `bill_id`, `text`, and `title` to focus only on the inputs and outputs required for model training.

---

## Training

### Training Setup
- **Batch Size**: 2 (to manage memory constraints).
- **Gradient Accumulation**: Steps set to 4 to simulate a larger batch size.
- **Learning Rate**: 5e-5.
- **Number of Epochs**: 3.
- **Optimizer**: AdamW with weight decay of 0.01.
- **Evaluation Strategy**: Performed at the end of each epoch.
- **Hardware**: Utilized GPU for training with CUDA enabled.

### Training Process
- The model was fine-tuned on the U.S. training dataset (`us_train_data_final_official.jsonl`) and evaluated using the U.S. test dataset (`us_test_data_final_official.jsonl`).

---

## Evaluation

### Metrics
The model's performance was evaluated using the Rouge metric:
- **Rouge-1**: Measures overlap of unigrams (words).
- **Rouge-2**: Measures overlap of bigrams (two consecutive words).
- **Rouge-L**: Measures longest common subsequences.

### Results
| Metric      | Score   |
|-------------|---------|
| **Rouge-1** | 0.2449  |
| **Rouge-2** | 0.1049  |
| **Rouge-L** | 0.1686  |

### Observations
- The model effectively captures key points in the summaries but can further improve with additional fine-tuning or data augmentation.

---

Feel free to explore the repository and adapt it for other summarization tasks. Contributions are welcome!
