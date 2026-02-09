import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("Central Limit Theorem Demonstration")

st.write(
    """
    This application demonstrates the Central Limit Theorem (CLT) by showing how
    the distribution of sample means converges to a normal distribution as sample
    size increases, even when the underlying data are not normally distributed.
    """
)

distribution = st.selectbox(
    "Choose a distribution",
    ["Exponential", "Uniform"]
)

sample_size = st.slider(
    "Sample size",
    min_value=1,
    max_value=500,
    value=50
)

simulations = st.slider(
    "Number of simulations",
    min_value=100,
    max_value=5000,
    value=1000
)

if distribution == "Exponential":
    data = np.random.exponential(scale=1.0, size=(simulations, sample_size))
    theoretical_mean = 1.0
    theoretical_std = 1.0 / np.sqrt(sample_size)
else:
    data = np.random.uniform(low=0.0, high=1.0, size=(simulations, sample_size))
    theoretical_mean = 0.5
    theoretical_std = (1 / np.sqrt(12)) / np.sqrt(sample_size)

sample_means = data.mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(sample_means, bins=30, density=True, alpha=0.7, label="Sample means")

x = np.linspace(sample_means.min(), sample_means.max(), 200)
ax.plot(
    x,
    norm.pdf(x, theoretical_mean, theoretical_std),
    linestyle="--",
    label="Normal approximation"
)

ax.set_xlabel("Sample mean")
ax.set_ylabel("Density")
ax.set_title("Distribution of Sample Means")
ax.legend()

st.pyplot(fig)

st.write(
    """
    **Interpretation**

    As the sample size increases, the distribution of sample means becomes
    increasingly normal, regardless of the shape of the underlying distribution.
    This property explains why normal-based statistical methods are often effective
    even when individual observations are skewed.
    """
)
