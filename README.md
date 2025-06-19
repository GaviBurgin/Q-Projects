# Q-Projects

This repository is for my **risk-neutral quantitative projects**. These techniques are mainly used in the valuation of derivatives and hedging strategies. As I gain experience, I'll continue adding projects in finance and related fields.

## 📘 Current Projects

## 1. Option Pricing under stochastic volatility
- Using Monte Carlo simulations to price call options under the Heston model.
- Exploring how changes in heston model parameters affect volatility surfaces.

## 2. Machine Learning Pricing of Barrier Options (Honors Research)
- Simulating Barrier option prices and arbitrage free volatility surfaces from stock path simulations under the Heston model.
- Developing a machine learning framework for pricing barrier options by mapping from an implied volatility surfaces and Barrier level to a Barrier price; (K,T,sigma,B) -> Barrier price.
- Model Validation.

## 3. Solving PDEs using Physics-Informed Neural Networks (PINNs)
- Heat equation
- Call option under Black-Scholes
  
## 4. Monte Carlo pricing excercise with varaince reduction 
- Using Monte Carlo simulations to price Fade-in call/put options under a GBM.
- Variance reduction techniques employed: Antithetic sampling, importance sampling, End-point stratification.
- Multiple control variates explored too.
- Explore trade-off in efficiency vs accuracy of pricing algorithm

## 5. Term-Structure Modelling: vasicek (1-factor) and AFNS (3-factor) (Masters Research project semester 1)
- Estimating parameters of short-rate models using a combinded kalman-filter and liklihood approach.
- Model evolution of short rate under Q and P measure
- Model Validation.
- Forecasting
  
## 🚀 Projects in the Pipeline

## Machine Learning Pricing of Barrier Options using Physics-Informed Neural Networks (PINNs) (Honors Research Extended)
- Extending the research on barrier options pricing by leveraging Physics-Informed Neural Networks (PINNs).
- Embedding the Heston risk-neutral PDE directly into the neural network architecture to fix some shortcomings of the previous model.
- Model Validation.


## 🛠 Tools I'm Using
- **Programming Language**: Python, Matlab
- **Techniques**:
  - Monte Carlo simulations
  - Risk-neutral pricing
  - Stochastic modelling
  - Machine Learning
  - variance reduction
- **Packages**:
  - Numpy
  - PyTorch
  - Matplotlib
  - SciPy
  - Sklearn
