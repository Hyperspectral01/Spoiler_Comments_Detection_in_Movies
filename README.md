
# 🎬 Context-Aware Movie Spoiler Classification 🤖

## Table of Contents

1.  [🌟 Project Overview](https://www.google.com/search?q=%23-project-overview)
2.  [✨ Model Architecture: The Four-Stage Pipeline](https://www.google.com/search?q=%23-model-architecture-the-four-stage-pipeline)
3.  [💻 Setup and Installation](https://www.google.com/search?q=%23-setup-and-installation)
4.  [🚀 How to Run the Project](https://www.google.com/search?q=%23-how-to-run-the-project)
      * [Training the Model (`train.py`)](https://www.google.com/search?q=%23training-the-model-trainpy)
      * [Custom Inference (`custom_test.py`)](https://www.google.com/search?q=%23custom-inference-custom_testpy)
5.  [📈 Training Performance and Optimization](https://www.google.com/search?q=%23-training-performance-and-optimization)
6.  [📂 Dataset Structure](https://www.google.com/search?q=%23-dataset-structure)
7.  [🛠️ Dependencies and Requirements](https://www.google.com/search?q=%23-dependencies-and-requirements)

-----

## 🌟 Project Overview

This project presents a novel, context-aware deep learning solution for classifying user comments as either a **spoiler** or **non-spoiler** for a specific movie. Unlike traditional sentiment or text classification, this task requires deep **contextual understanding**, as a comment's spoiler status is entirely dependent on the movie's plot.

The core innovation is a **four-stage neural network pipeline** that simultaneously processes a full movie's subtitles and a user's comment, using a sophisticated **Attention Mechanism** to find the exact plot context relevant to the comment before making a final prediction.

### Key Achievements

  * **Best Validation Accuracy:** **95.87%** (achieved at Epoch 7)
  * **Proactive Solution:** Provides an automated alternative to manual spoiler flagging, which is often slow and reactive.
  * **Contextual Deep Learning:** Successfully integrates subtitle-based movie context into the classification decision.

-----

## ✨ Model Architecture: The Four-Stage Pipeline

The solution is implemented as a pipeline of four distinct PyTorch models (`Model1` through `Model4`), leveraging a pre-trained feature extractor (specifically **allenai/scibert\_scivocab\_uncased**) to handle textual data effectively.

[Image of Deep Learning Model Flowchart]

The data flow can be visualized as a structured pipeline:

### 1\. Model 1: Subtitle Processor (Context Encoder)

  * **Purpose:** To encode the complete narrative of the movie's subtitles (`.srt` file) into a fixed sequence of rich hidden states.
  * **Architecture:** The entire subtitle dialogue is **chunked into 20 equal-sized segments** (the number is variable). Each segment's word embeddings are processed by a **GRU** (Gated Recurrent Unit).
  * **Output:** A tensor of shape **[20, 1024]** (20 hidden states, each summarizing a part of the movie).

### 2\. Model 2: Comment Processor (Comment Encoder)

  * **Purpose:** To transform each user comment into a single, dense vector representation.
  * **Architecture:** Each comment is passed through the same SciBERT feature extractor, padded/truncated to a uniform length (max 256 tokens in `train.py`). The sequence of comment embeddings is then processed by a separate **GRU**.
  * **Output:** A tensor of shape **[Batch Size, 768]**, where 768 is the size of the GRU's final hidden state for the comment.

### 3\. Model 3: Attention Mechanism (Context Selector)

  * **Purpose:** The critical step that determines which parts of the movie are most relevant to the comment.
  * **Mechanism:** Uses a simple **Dot Product Attention** mechanism. It calculates similarity scores between the single comment vector (from Model 2) and all 20 movie chunks (from Model 1).
  * **Selection:** The model uses **top-K selection ($k=2$)** (k is variable) to identify the two most relevant subtitle hidden states for that specific comment.
  * **Output:** A tensor of shape **[Batch Size, 2, 1024]**, which represents the two most pertinent contexts for each comment.

### 4\. Model 4: Final Classifier (Prediction Head)

  * **Purpose:** To make the final binary prediction (Spoiler / Non-Spoiler).
  * **Architecture:** The single comment vector (768-dim) and the two selected context vectors (flattened to $2 \times 1024 = 2048$-dim) are **concatenated**. This combined vector is fed through a multi-layer fully-connected network with **Batch Normalization** and **Dropout** for regularization.
  * **Output:** A tensor of shape **[Batch Size, 2]**, providing the raw logits for:
      * **Index 0:** Non-Spoiler
      * **Index 1:** Spoiler

-----

## 💻 Setup and Installation

### Hardware Requirements

Training deep learning models like this requires a **GPU (Graphics Processing Unit)** for efficient computation.

### Dependencies

You need to install the required Python packages using `pip`.

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib
```

### Pretrained Model

The project uses the `allenai/scibert_scivocab_uncased` as a text embedding model. For covenience, I have included the text-embedding model in this repo, and the files train.py and custom_test.py automatically uses the embedding model requiring no extra steps to run anything.

-----

## 🚀 How to Run the Project

### Training the Model (`train.py`)

The `train.py` script handles the entire training process, including data loading, forward/backward passes, optimization, evaluation, and saving the best-performing models with early stopping.

**Inputs:**

  * `train_dir_path="./train"`: Directory containing training data (`.srt` and corresponding `.csv` files).
  * `valid_dir_path="./valid"`: Directory containing validation data.

**Key Training Optimizations:**

  * **Adam Optimizer:** Used for all four models with a learning rate of $0.0005$.
  * **L2 Regularization (Weight Decay):** Applied to the loss function to prevent overfitting, with a $\lambda$ value of $0.001$.
  * **Gradient Clipping:** Used to prevent exploding gradients during training.
  * **Learning Rate Scheduler:** `ReduceLROnPlateau` is used on the validation loss (`scheduler.step(val_loss)`), which reduces the learning rate by a factor of 0.5 if the loss plateaus for 3 epochs.
  * **Early Stopping:** Training terminates if the validation accuracy shows no improvement for **5 consecutive epochs** of patience.

To start the training:

```bash
python train.py
```

The model weights will be saved to the `./out` directory.

### Custom Inference (`custom_test.py`)

This script allows you to load the best-performing model weights and test them on new, custom comments for an unseen movie.

1.  **Ensure Best Models are Saved:** After running `train.py`, the best weights (e.g., `best_model1_weights.pth`) must be in the `out/` directory.
2.  **Update Inputs:** Modify the `custom_test.py` file's inputs section:
      * `srt_file_path`: Point this to the full path of your test movie's subtitle file.
      * `sample_comments`: Edit this list with the custom comments you want to classify.

The script will load the weights, run the inference pipeline, and print the classification for each comment.

```bash
python custom_test.py
```

-----

## 📈 Training Performance and Optimization

The project demonstrated excellent convergence and high peak performance:

| Metric | Value | Epoch | Context |
| :--- | :--- | :--- | :--- |
| **Peak Validation Accuracy** | **95.87%** | **7** | The highest generalization achieved. |
| Final Validation Accuracy | 94.83% | 12 | The final achieved accuracy before early stopping. |
| Total Train Samples | 4,488 | - | Divided across 38 movies. |
| Total Validation Samples | 484 | - | Divided across 3 movies. |

The training logs show the early stopping mechanism successfully halted training at **Epoch 12** after the validation accuracy failed to improve for 5 consecutive epochs (after Epoch 7's peak).

The training curves visually confirm this behavior: the **Training Loss** continues to decrease while the **Validation Accuracy** plateaus and becomes volatile after the peak, demonstrating successful use of early stopping to prevent severe overfitting.

-----

## 📂 Dataset Structure

The training and validation directories should contain pairs of subtitle and comment files.

  * The subtitle file must have the `.srt` extension.
  * The comments file must have the same name as the subtitle file but with a **`.csv`** extension.
  * The comment files must be formatted as CSVs with the first column being the label (`spoiler` or `non-spoiler`) and the second column being the comment text.

<!-- end list -->

```
.
├── train/
│   ├── MovieA.srt
│   ├── MovieA.csv
│   ├── MovieB.srt
│   └── MovieB.csv
└── valid/
    ├── MovieY.srt
    ├── MovieY.csv
    └── ...
```