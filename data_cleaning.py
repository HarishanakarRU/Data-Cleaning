import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(0)
plt.style.use('fivethirtyeight')
netflix = pd.read_csv("netflix1.csv")
print(netflix.head())
print(netflix.columns)
print(netflix["rating"])

# No missing values each column
print(sum([sum(netflix[column].isna()) for column in netflix.columns])) # Equals 0

# Examine Column Types of Dataset
column_types = {}

for column in netflix.columns:
    column_types[column] = str(netflix[column].dtype)
print(column_types)

# EDA - TV Rating Distribution
counts = netflix["rating"].value_counts()

# plt.bar(counts.index, counts)



# Mapping Duration to Time

# Movies
movies = netflix[netflix["type"] == "Movie"]
print(movies.head())

# To map duration to time, we need to split the string and convert the first element to int
movies["length"] = movies["duration"].str.split().str[0]
durations = movies["length"].value_counts()

# Statistical Inference

'''
Let X_1, X_2, ... , X_n be iid Exp(lambda). Suppose durations is the set of datapointswhich represent this model, we can estimate:
- MLE since we have a parametric model. My reasoning for choosing exponential stems from the EDA whose distribution looks similar to
Exp(lambda)
'''

lambda_hat = 1 / np.mean(durations) # Derived from MLE calculation
print(f"MLE lambda: {lambda_hat}")

# Exp(lambda_hat) Plot with Durations distribution

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 160, 500)
exp_pdf = stats.expon.pdf(x, 0, 1/lambda_hat)

ax.hist(durations, density = True, label = "Duration Distribution (from data)")
ax.plot(x, exp_pdf, '--', color = 'firebrick', linewidth = 2.5,
        label = f'Exp({lambda_hat:.5f})')

ax.set_xlabel('Durations (Minutes)', fontsize = 12)
ax.set_ylabel('Density', fontsize = 12)
ax.set_title('Durations Distribution with Exp($\hat{\lambda}$) Model', fontsize = 14, fontweight = 'bold')
ax.legend(fontsize = 11)

plt.show()

# Parametric Bootstrap
B = 10000
lambda_hats = []
n = len(durations)

for i in range(B):
    # Draw from Exp distribution
    random_samples = np.random.exponential(1 / lambda_hat, n)

    # Calculate MLE
    lambda_star = 1 / np.mean(random_samples)

    # Add to list
    lambda_hats.append(lambda_star)



# Delta Method Approximation N(1/lambda_hat, 1/lambda_hat**2 * n)
x = np.linspace(min(lambda_hats), max(lambda_hats), 100)
delta_pdf = stats.norm.pdf(x, lambda_hat, np.sqrt((lambda_hat**2)/ n))

# Plots
fig, ax = plt.subplots(figsize=(10, 6))


ax.hist(lambda_hats, density = True, label = "Parametric Bootstrap Distribution")
ax.plot(x, delta_pdf, '--', color = 'firebrick', linewidth = 2.5,
        label = f'Delta method: N({lambda_hat:.5f}, {np.sqrt((lambda_hat**2)/ n):.5f}²)')
ax.axvline(lambda_hat, color = 'black', linestyle = '--', linewidth = 1.5, alpha = 0.7,
           label = f'MLE estimate: λ_hat = {lambda_hat:.5f}')

ax.set_xlabel('Estimated Mean (Duration/Movie)', fontsize = 12)
ax.set_ylabel('Density', fontsize = 12)
ax.set_title('Bootstrap vs Delta Method Prediction for $\hat{\lambda}$', fontsize = 14, fontweight = 'bold')
ax.legend(fontsize = 11)

plt.tight_layout()
plt.show()




