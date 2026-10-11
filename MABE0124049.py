import numpy as np
import pandas as pd

df = pd.read_csv("Ads_CTR_Optimisation.csv")
print(df.describe())

num_users, num_ads = df.shape

counts = np.zeros(num_ads)
rewards = np.zeros(num_ads)
epsilon = 0.1

for t in range(num_users):
    if np.random.rand() < epsilon:
        ad = np.random.randint(num_ads)
    else:
        avg_rewards = np.zeros(num_ads)
        for i in range(num_ads):
            if counts[i] > 0:
                avg_rewards[i] = rewards[i] / counts[i]
            else:
                avg_rewards[i] = 0
        ad = np.argmax(avg_rewards)
        
    click = df.iloc[t, ad]
    counts[ad] += 1
    rewards[ad] += click

best_ad = np.argmax(rewards / np.where(counts == 0, 1, counts))

print("\nTimes Selected:", counts)
print("Total Clicks:", rewards)
print("Best Performing Ad:", best_ad)
