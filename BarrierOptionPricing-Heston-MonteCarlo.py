import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def simulate_multiple_paths_heston(delta_t, r, kappa, theta, xi, rho, s_0, v_0, n_iterations, n_paths):
  # Generate time steps
  end_time = n_iterations * delta_t
  num_points = n_iterations + 1  # Include the endpoint
  times = np.linspace(0, end_time, num_points)

  # Covariance matrix with correlation rho
  cov_matrix = [[delta_t, rho * delta_t],
                [rho * delta_t, delta_t]]

  # Sample from a multivariate normal distribution
  correlated_random_walks = np.random.multivariate_normal([0, 0], cov_matrix, (n_paths, n_iterations))

  # Extract the individual Wiener processes
  dW_s = correlated_random_walks[:, :, 0]
  dW_v = correlated_random_walks[:, :, 1]

  # Initialize array for volatilities
  volatilities = np.zeros((n_paths, n_iterations + 1))
  volatilities[:, 0] = v_0

  # Vectorized calculation of volatility paths
  v_prev = np.full((n_paths, n_iterations), v_0)
  volatilities[:, 1:] = np.maximum(v_prev + kappa * (theta - v_prev) * delta_t + xi * np.sqrt(v_prev) * dW_v, 0)

  # Initialize array for prices
  prices = np.zeros((n_paths, n_iterations + 1))
  prices[:, 0] = s_0

  # Vectorized calculation of price paths
  s_prev = np.full((n_paths, n_iterations), s_0)
  prices[:, 1:] = s_prev * np.cumprod(1 + r * delta_t + np.sqrt(volatilities[:, :-1]) * dW_s, axis=1)

  return times, prices



def option_price_calc(times,paths,T,t,r,k):

  n_paths = paths.shape[0]
  final_payoffs = np.zeros(n_paths)
  time_position = np.argmin(np.abs(times - T))
  # Directly get the final price for all paths at the stop time
  final_prices = paths[:, time_position]#time_position
  final_payoffs = np.maximum(final_prices - k, 0)

  # Discount the average of the payoffs back to present value
  average_payoff = np.mean(final_payoffs)
  discounted_payoff = np.exp(-r * (T-t)) * average_payoff

  return discounted_payoff


def check_martingale(times,paths,T,t,r,k):

  n_paths = paths.shape[0]
  final_payoffs = np.zeros(n_paths)
  time_position = np.argmin(np.abs(times - T))

  # Directly get the final price for all paths at the stop time
  final_prices = paths[:, time_position]#time_position

  # Discount the average of the payoffs back to present value
  average_payoff = np.mean(final_prices)
  discounted_payoff = np.exp(-r * (T-t)) * average_payoff

  return discounted_payoff


def barrier_upAndIn_price_calc(times, paths, T, t, r, k, barrier):
  
  # Determine the position in the times array corresponding to stop_time
  time_position = np.argmin(np.abs(times - T))

  # Calculate the maximum price for each path up to the stop time
  max_prices_up_to_stop = np.max(paths[:, :time_position + 1], axis=1)

  # Identify paths where the maximum price crossed the barrier
  crossed_barrier = max_prices_up_to_stop > barrier

  # Get final prices at the stop time for paths
  final_prices_at_stop = paths[:, time_position]

  # Calculate the payoff for each path that crossed the barrier
  payoffs = np.where(crossed_barrier, np.maximum(final_prices_at_stop - k, 0), 0)

  # Calculate the average of the payoffs and discount it to the present value
  average_payoff = np.mean(payoffs)
  discounted_payoff = np.exp(-r * (T - t)) * average_payoff

  # Identify paths that did cross the barrier
  crossed_paths = paths[crossed_barrier]

  # Return both the discounted payoff and the paths that did not cross the barrier
  return discounted_payoff, crossed_paths

def plot(times, paths, kappa, theta, xi, rho):
  
  plt.plot(times, paths.T)
  plt.title(f'Simulated Stock Price Paths Over Time ({paths.shape[0]} paths)\n' \
            f'kappa={kappa}, theta={theta}, xi={xi}, rho={rho}')
  plt.xlabel('Time in years')
  plt.ylabel('Stock Price')
  plt.grid(True)
  plt.show()

#heston params
r=0.08
v_0 = 0.2 # initial vol

kappa = 3
theta = 0.5
xi = 0.3
rho=-0.2

T=1.5
s_0 = 1
K=120
t = 0
B = 140

#sim params
delta_t = 1/365
n_iterations = int(T / delta_t)
n_paths = 500

times, price_paths = simulate_multiple_paths_heston(delta_t, r, kappa, theta, xi,rho,s_0, v_0**2, n_iterations, n_paths)
plot(times,price_paths,kappa, theta, xi, rho)

option_price = option_price_calc(times,price_paths,T,t,r,K)
mart = check_martingale(times,price_paths,T,t,r,K)
barrierUI, barrier_paths =barrier_upAndIn_price_calc(times, price_paths, T, t, r, K, B)


print(f"Simulated Heston Call Price:        R{option_price:5.2f}")
print(f"check if martingale:                R{mart:5.2f}")
print(f"Simulated Heston up and in Price:   R{barrierUI:5.2f}")


