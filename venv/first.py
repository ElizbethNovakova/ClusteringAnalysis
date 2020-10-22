import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


list = np.random.randint(0, 100, size=(1000, 2))
df = pd.DataFrame(list, columns=['X', 'Y'])
print(df)

km = KMeans(n_clusters=2).fit(df)
centr = km.cluster_centers_
labels = km.labels_
df['Cluster'] = labels

print(centr)
print(labels)


clusters_group = df.groupby('Cluster')
print("\nDescribe: \n", clusters_group.describe().T)


plt.scatter(df.X, df.Y, c= labels.astype(float), s=20, alpha=0.5)
plt.scatter(centr[:, 0], centr[:, 1], c='red', s=50)
plt.show()

