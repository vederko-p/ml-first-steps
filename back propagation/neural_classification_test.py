
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


classification_data = pd.read_csv('data/classification_two_clusters.csv')
class_val = classification_data.values

x1, x2, y = class_val[:, 1].reshape(-1, 1), class_val[:, 0].reshape(-1, 1), class_val[:, 2]

fig, ax = plt.subplots()
ax.scatter(x1, x2)
plt.show()


print(y)
