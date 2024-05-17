# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:41:24 2024

@author: Dominic Beler
"""

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns

sns.set(style="darkgrid", palette="muted")

def simulate_linear_data(
    start, stop, N, beta_0, beta_1, eps_mean, eps_sigma_sq
):

    # Generate X values
    X = np.linspace(start, stop, N)
    # Generate epsilon (noise)
    eps = np.random.normal(loc=eps_mean, scale=np.sqrt(eps_sigma_sq), size=N)
    # Generate y values
    y = beta_0 + beta_1 * X + eps
    return X, y

# Define true parameters
true_beta_0 = 2.0
true_beta_1 = 3.0
true_eps_mean = 0.0
true_eps_sigma_sq = 1.0

# Simulate data
np.random.seed(0)  # For reproducibility
start = 0
stop = 10
N = 50
X, y = simulate_linear_data(
    start, stop, N, true_beta_0, true_beta_1, true_eps_mean, true_eps_sigma_sq
)

# Plot simulated data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=30, label="Simulated data")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simulated Data")
plt.legend()
plt.show()

# Bayesian Linear Regression Model
with pm.Model() as linear_model:
    # Priors for unknown model parameters
    beta_0 = pm.Normal("beta_0", mu=0, sigma=10)
    beta_1 = pm.Normal("beta_1", mu=0, sigma=10)
    eps_sigma = pm.HalfNormal("eps_sigma", sigma=1)

    # Expected value of outcome
    mu = beta_0 + beta_1 * X

    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal("y_obs", mu=mu, sigma=eps_sigma, observed=y)

    # Bayesian Inference
    trace = pm.sample(1000, tune=1000)

# Plot the trace
pm.traceplot(trace)
plt.show()

# Summarize the posterior distribution
print(pm.summary(trace))

# Forecasting
X_new = np.linspace(0, 12, 100)  # New X values for forecasting
with linear_model:
    # Generate posterior predictive samples
    post_pred = pm.sample_posterior_predictive(trace, samples=1000)

# Plot posterior predictive distribution
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=30, label="Simulated data")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Posterior Predictive Distribution")
plt.plot(X_new, post_pred["y_obs"].T, "b-", alpha=0.1)
plt.show()

