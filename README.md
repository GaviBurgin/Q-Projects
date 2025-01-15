# Q-Projects

This repository is for my **risk-neutral quantitative projects**. These techniques are widely used in the valuation of derivatives, hedging strategies, market risk assessment, and ensuring arbitrage-free pricing in financial models. As I gain experience, I'll continue adding projects in finance and related fields.

## 📘 Current Project

## 1. Barrier Option Pricing under stochastic volatility
- see: BarrierOptionPricing-Heston-MonteCarlo.py
- Using Monte Carlo simulations to price barrier options under the Heston model.
- Exploring how changes in model parameters affect option prices (to be uloaded soon).

## 2. Machine Learning Pricing of Barrier Options (Honors Research)
- see: NN_w_constraints.ipynb
- Generating arbitrage free volatility surfaces from stock path simulations under the Heston model (to be uloaded soon).
- Developing a machine learning framework for pricing barrier options by mapping from an implied volatility surfaces and Barrier level to a Barrier price; (K,T,sigma,B) -> Barrier price.
- Model Validation.

## 🚀 Projects in the Pipeline

## Machine Learning Pricing of Barrier Options using Physics-Informed Neural Networks (PINNs) (Honors Research Extended)
- Extending the research on barrier options pricing by leveraging Physics-Informed Neural Networks (PINNs).
- Embedding the Heston risk-neutral PDE directly into the neural network architecture to fix some shortcomings of the previous model.
- Model Validation.


## 🛠 Tools I'm Using
- **Programming Language**: Python
- **Techniques**:
  - Monte Carlo simulations
  - Risk-neutral pricing
  - Stochastic processes (Heston model)
  - Machine Learning
- **Packages**:
  - Numpy
  - PyTorch
  - Matplotlib
  - SciPy
  - Sklearn
