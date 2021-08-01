import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

arr = np.random.randint(0, 10, 25).reshape(5, 5)

cmap = colors.ListedColormap(
    ["navy", "royalblue", "lightsteelblue", "beige", "peachpuff"]
)

bounds = [2, 4, 6, 8, 10]


fig, ax = plt.subplots()
# im = ax.imshow(arr, cmap=cmap)

# fig.show()


import seaborn as sns

ax = sns.heatmap(arr, cmap=cmap)
fig.show()
