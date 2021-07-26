
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def true_func(x):
    return 3.5*x + 8


x = np.append(np.linspace(0, 10, 30), np.linspace(11, 30, 30))
y = true_func(x) + np.random.normal(scale=9, size=x.size)

true_x = np.linspace(0, 30, 100)
true_y = true_func(true_x)

'''
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x, y, linewidths=0.1)
ax.plot(true_x, true_y, c='orange')
plt.show()
'''

'''
points = np.array([y, x]).transpose()
points_df = pd.DataFrame(points, columns=['y', 'x'])
points_df.to_csv('data/delete_it.csv', index=False)
'''


o = np.ones(20)
class_x = np.append(o + 1 + np.random.normal(scale=3, size=20),
                    o + 20 + np.random.normal(scale=3, size=20))
class_y = np.append(o + 1 + np.random.normal(scale=3, size=20),
                    o + 20 + np.random.normal(scale=3, size=20))


fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(class_x, class_y, linewidths=0.1)
plt.show()


'''
classes = np.ones(40).reshape(-1, 1)
classes[:20] = -1 * classes[:20]
points = np.append(np.array([class_y, class_x]).transpose(), classes, axis=1)
points_df = pd.DataFrame(points, columns=['y', 'x', 'class'])
points_df.to_csv('data/delete_it.csv', index=False)
'''
