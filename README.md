<div align="center">

# Core AI Algorithms

**A structured learning path through classical ML, deep learning, computer vision, and soft computing, with algorithms implemented from scratch and compared against library versions.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-learning%20repo-lightgrey)

</div>

---

## About

This repository is my hands-on notebook for understanding how core AI algorithms actually work. Each module pairs a short conceptual explanation with runnable code: where it teaches the most, I implement the algorithm from scratch, then compare it with the standard library implementation.

It is a **learning repository**, not a production system. The goal is depth of understanding: the math, the mechanics, and how to read the results.

## Curriculum

| # | Module | What it covers | Key techniques |
|---|--------|----------------|----------------|
| 01 | [Classical ML](01_classical_ml/) | Optimization, trees, and the boosting family | Gradient Descent / SGD, Decision Tree, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, loss functions |
| 02 | [Neural Networks](02_neural_networks/) | Core deep learning architectures, built step by step | MLP, CNN, RNN, Transformer |
| 03 | [Computer Vision](03_computer_vision/) | Classical image processing | Filtering, noise, morphological operations, edge detection and gradients |
| 04 | [Soft Computing](04_soft_computing/) | Fuzzy and evolutionary methods | Fuzzy C-Means, LVQ, genetic algorithms, fuzzy control |
| 05 | [Model Interpretation](05_model_interpretation/) | A visual handbook for reading model diagnostics | ROC, Precision-Recall, SHAP, learning curves |

## Repository Structure

```text
Core_ai_algos/
├── 01_classical_ml/          # optimization, trees, boosting family
├── 02_neural_networks/
│   ├── 01_mlp/
│   ├── 02_cnn/
│   ├── 03_rnn/
│   └── 04_transformers/
├── 03_computer_vision/       # image processing notebooks
├── 04_soft_computing/        # fuzzy systems and genetic optimization
├── 05_model_interpretation/  # diagnostics guide
├── LICENSE
└── README.md
```

## Module Details

### 01 · Classical ML
Starts from the optimization basics (gradient descent and SGD) and builds up to tree-based ensembles.
- **From scratch:** Decision Tree, Gradient Boosting, AdaBoost, and a vectorized NumPy loss-function library with gradients.
- **Library implementations:** XGBoost, LightGBM, CatBoost, used to compare against the scratch versions.

### 02 · Neural Networks
The four architectures that most of modern deep learning builds on, each implemented from scratch: **MLP → CNN → RNN → Transformer**. Every folder walks through the forward pass, the backward pass (where applicable), and training.

### 03 · Computer Vision
Notebooks on the fundamentals of image processing: image basics, noise and filtering, morphological operations, and edge detection with gradients.

### 04 · Soft Computing
Non-neural intelligent systems: Fuzzy C-Means clustering, Learning Vector Quantization, genetic algorithm optimization, and fuzzy control.

### 05 · Model Interpretation
A practical guide to reading the plots that come out of a model: ROC and Precision-Recall curves, SHAP explanations, and learning curves for spotting over- and underfitting.

## Getting Started

```bash
# 1. Clone
git clone https://github.com/laila-kz/Core_ai_algos.git
cd Core_ai_algos

# 2. Create an environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Explore
jupyter lab
```

Each module folder can be run on its own. Open its notebook, or run its scripts directly, for example `python 01_classical_ml/<script>.py`.

## Tech Stack

Python · NumPy · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · OpenCV · scikit-image · matplotlib · SHAP · Jupyter

## What I Learned

- Implementing an algorithm from scratch exposes details that library calls hide, such as how splits are chosen, how gradients flow, and why numerical stability matters.
- Comparing a scratch version against a library version is the fastest way to find bugs in your own understanding.
- Being able to read diagnostic plots matters as much as training the model.

## Author

**Leila Khezaz**: Final-year AI & Data Science engineering student at ENSA Safi, focused on Data Engineering and Analytics Engineering.

[GitHub](https://github.com/laila-kz) · [LinkedIn](https://linkedin.com/in/leila-khezaz-a57779336) · leilakhezaz07@gmail.com

## License

Released under the [MIT License](LICENSE).