<div align="center">
  <h1>GANerator</h1>
  <h3>An MVP for applying Machine Learning (ML) in the credit risk modeling lifecycle. </h1>
</div>

It currently contains two types of models:

1. 🎯 **Credit Stress Predictor**
  - ML Method: Random Forest - A machine learning algorithm that combines multiple decision trees to create a singular, more accurate result.
2. 🔄 **Data Synthesizer**
  - ML Method: GAN (Generative Adversarial Network) - A deep learning method in which two neural networks compete with each other in a game, learning to generate new data with the same statistics as the training set



## Installation

> **NOTE**: The whole data set can be downloaded from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv).
> Store the files _application_test.csv_ and _application_train.csv_ in the [/data](/data) directory.

[![PyTorch](https://skillicons.dev/icons?i=pytorch)](https://pytorch.org/)
[![SkLearn](https://skillicons.dev/icons?i=sklearn)](https://scikit-learn.org/)
[![Python](https://skillicons.dev/icons?i=py)](https://python.org/)

Create a new virtual environment (venv) and activate it:

    python -m venv venv
    source venv/bin/activate

Install requirements and the repo itself:

    pip install -r requirements.txt
    pip install -e .

Create a new jupyter kernel:

    pip install ipykernel
    python -m ipykernel install --user --name ganerator-dev


## Usage

1. **Credit Stress Predictor** - Run the [Risk Modelling Notebook](notebooks/risk_modelling.ipynb).

2. **Data Synthesizer** - Run the [Data Synthesizer Notebook](notebooks/data_synthesizer.ipynb).
