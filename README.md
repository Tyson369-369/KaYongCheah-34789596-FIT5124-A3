This project implements and evaluates model extraction (MEA) and membership inference (MIA) attacks and their corresponding defences on an MNIST digit-classification model.


# Model Extraction & Membership Inference on MNIST

This repository explores security and privacy risks of machine learning APIs by implementing **Model Extraction Attacks (MEA)** and **Membership Inference Attacks (MIA)** against a LeNet-style CNN trained on MNIST, together with practical output-level defences.

## Project overview

The work is organised into four main components:

- Training a **target MNIST classifier** (LeNet-style CNN).  
- Designing and evaluating a **black-box model extraction attack** and a **label-only defence**.  
- Designing and evaluating a **membership inference attack** and a **noise-based Softmax-flattening defence**.  
- Analysing how query budget, training configuration, and API output format affect attack success and defence effectiveness.

## Files

- `a3_mnist.py`  
  - Trains a LeNet-style convolutional network on MNIST.  
  - Uses Adadelta with learning rate 1 and a StepLR scheduler over 25 epochs.  
  - Saves the trained model weights to `target_model_25.pth`.

- `mea_attack_task3.py`  
  - Simulates a black-box API that wraps the target model and returns full log-softmax scores.  
  - Implements a 3-layer fully connected surrogate (“Attack” model) trained with MSE loss on query–response pairs.  
  - Uses 30% of the MNIST test set (3,000 queries) to perform a model extraction attack and reports extraction accuracy.

- `mea_defence_task4.py`  
  - Reuses the same target and surrogate architectures.  
  - Implements a **label-only defence** that replaces full confidence scores with log-transformed one-hot vectors for the predicted label.  
  - Runs extraction against both the original and defended APIs, reporting extraction accuracy, average training loss, and runtime.

- `mia_attack__task5.py`  
  - Loads the trained target model and keeps it fixed.  
  - Treats MNIST training samples as **members** and EMNIST `digits` test samples as **non-members**.  
  - Collects Softmax outputs from the target model and trains a logistic regression attack model to infer membership from probability vectors.  

- `mia_defence__task6.py`  
  - Implements a **noise-based Softmax-flattening defence**: replaces the model’s original Softmax scores with a nearly uniform, noisy distribution that only mildly prefers the predicted class.  
  - Uses these defended outputs as features for a logistic regression MIA, and evaluates how much the defence reduces membership inference performance and runtime.

## Setup

1. **Create and activate a virtual environment (optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   # venv\Scripts\activate       # Windows
   

2. **Install dependencies**

   ```bash
   pip install torch torchvision scikit-learn
   ```

3. **Datasets**

   - MNIST and EMNIST (`digits` split) are downloaded automatically to `./data` when the scripts are first run.

## 1. Train the target MNIST model

Train the baseline LeNet-style model on MNIST and save the weights:

```bash
python a3_mnist.py
```

This script:

- Normalises MNIST images with mean 0.1307 and std 0.3081.  
- Trains on 60,000 MNIST training images and evaluates on 10,000 test images.  
- Saves the trained model as `target_model_25.pth`.

If required by other scripts, copy or rename:

```bash
cp target_model_25.pth target_model.pth
```

so that a file named `target_model.pth` is present in the working directory.

## 2. Model Extraction Attack (MEA)

### 2.1 Attack: full-confidence black-box API

Run the model extraction attack:

```bash
python mea_attack_task3.py
```

Implementation highlights:

- A simulated black-box function loads the hidden target model and returns log-softmax outputs for each query.  
- The attacker defines an independent surrogate network (`Attack`), a 3-layer MLP (784 → 128 → 64 → 10) with ReLU activations and log-softmax output.  
- The attacker uses 3,000 queries (30% of the MNIST test set) and trains the surrogate with MSE loss to match the target’s confidence distribution.  
- Extraction accuracy is measured as the percentage of test inputs on which the surrogate and target predict the same class.

**Key MEA result**

- Under a 3,000-query budget and 10 training epochs, the extraction attack achieves **89.66% precision/accuracy**, indicating substantial behavioural cloning and data leakage from the unprotected API.

### 2.2 Defence: label-only API

Run extraction on both original and label-only defended APIs:

```bash
python mea_defence_task4.py
```

Defence design:

- The defended query function first computes the target’s prediction, then replaces the full Softmax vector with a one-hot vector for the predicted class.  
- A small constant is added and the vector is log-transformed to match the original log-softmax shape, but no confidence information is revealed beyond the predicted label.  
- The same surrogate architecture and training procedure are used, allowing direct comparison with the baseline attack.

**Key MEA defence results**

- Extraction accuracy on the original (full-softmax) API: **89.70%** under 3,000 queries.  
- Extraction accuracy on the defended label-only API: **67.44%** under the same 3,000-query budget.  
- Runtime remains around **3.10 seconds** in both settings, showing that the defence is lightweight and does not introduce noticeable computational overhead.

These results show that label-only output significantly weakens model extraction while preserving normal classification functionality.

## 3. Membership Inference Attack (MIA)

### 3.1 Attack: probability-based MIA

Run the membership inference attack:

```bash
python mia_attack__task5.py
```

Attack setup:

- **Members (label 1)**: random subset of 10,000 samples from the MNIST training set.  
- **Non-members (label 0)**: random subset of 10,000 samples from the EMNIST `digits` test split.  
- For each sample, the fixed target model’s Softmax output is used as a feature vector; a logistic regression classifier (`max_iter=1000`) is trained to distinguish members from non-members.  

**Key MIA result**

- The attack achieves **69.27% precision/accuracy**, demonstrating that the model’s probability outputs leak enough information to infer membership significantly better than random guessing (50%).

### 3.2 Defence: noise-based Softmax flattening

Run the defended MIA:

```bash
python mia_defence__task6.py
```

Defence design:

- The defence function still queries the target model but then discards the original Softmax outputs.  
- For each sample, it generates a nearly uniform probability vector with entries drawn from a narrow range (e.g., around 0.09–0.10) and slightly biases the top-1 predicted class, followed by normalisation.  
- These noisy, flattened distributions are used as features for the logistic regression membership classifier.

**Key MIA defence results**

- With the noise-based Softmax-flattening defence, membership inference accuracy drops to **49.53%**, close to random guessing.  
- Defence runtime remains **under 1 second**, and the target model’s utility for normal classification is preserved because predictions (argmax) are unchanged.

## Summary of main results

- **Model Extraction Attack (MEA)**  
  - Designed and executed a black-box MEA achieving **89.66% precision** on an unprotected model, demonstrating significant leakage through full confidence scores.  

- **Label-only defence for MEA**  
  - Implemented a label-only defence that **reduced extraction precision to 67.44%** under the same query budget, substantially increasing the attacker’s cost while maintaining normal API behaviour.  

- **Membership Inference Attack (MIA)**  
  - Designed and executed a membership inference attack achieving **69.27% precision**, showing the ability to infer training membership much better than random guessing.  

- **Noise-based Softmax-flattening defence for MIA**  
  - Implemented and evaluated a noise-based Softmax flattening defence that **reduced MIA accuracy to 49.53% (near random guessing)** while preserving model utility and keeping **runtime below 1 second**.

## Reproducibility notes

- All random sampling (e.g., subset selection from MNIST/EMNIST and data loader shuffling) can cause minor run-to-run variation in metrics.  
- For exact replication of reported numbers, set fixed random seeds and run on similar hardware, or reuse saved models and indices if provided.

