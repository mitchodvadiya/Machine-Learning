# Machine Learning Practicals

Python implementations of the 14 Machine Learning practical topics shown in the classroom practical list.

## Experiments

1. Data Analysis
2. Single Variable Linear Regression
3. Multiple Variable Linear Regression
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)
7. Random Forest
8. K-Means Clustering
9. Hierarchical Clustering
10. DBSCAN Clustering
11. Principal Component Analysis (PCA)
12. Kernel-Based Image Filtering
13. Convolutional Neural Network (CNN)
14. Recurrent Neural Network (RNN) using LSTM

## Setup

Use Python 3.10 or 3.11 for the smoothest compatibility with TensorFlow.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment:

```bash
python experiments/01_data_analysis.py
```

Change the filename to run another experiment.

## Notes

- Classical ML experiments use built-in or generated sample datasets.
- CNN uses the MNIST handwritten-digit dataset. TensorFlow downloads it on the first run.
- LSTM uses a generated sine-wave sequence, so no external dataset is required.
- The image-filtering experiment uses a sample image bundled with scikit-image.
- Deep-learning experiments may take longer and require a compatible TensorFlow installation.
