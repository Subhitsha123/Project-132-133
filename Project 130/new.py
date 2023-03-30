import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

df = pd.read_csv("main.csv")
X = np.array(list(zip(list(df["Mass"]),list(df["Radius"])))).reshape(len(list(df["Mass"])),2)

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

mass.sort()
radius.sort()
gravity.sort()

plt.plot(radius,mass)

plt.title("RADIUS VS MASS")
plt.xlabel("RADIUS")
plt.ylabel("MASS")
plt.show()

plt.plot(mass,gravity)

plt.title("MASS VS GRAVITY")
plt.xlabel("MASS")
plt.ylabel("GRAVITY")
plt.show()

plt.scatter(radius,mass)
plt.xlabel("RADIUS")
plt.ylabel("MASS")
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

cluster_1_x = []
cluster_1_y = []
cluster_2_x = []
cluster_2_y = []
cluster_3_x = []
cluster_3_y = []
cluster_4_x = []
cluster_4_y = []

for index, data in enumerate(X):
  if y_kmeans[index] == 0:
    cluster_1_x.append(data[0])
    cluster_1_y.append(data[1])
  elif y_kmeans[index] == 1:
    cluster_2_x.append(data[0])
    cluster_2_y.append(data[1])
  elif y_kmeans[index] == 2:
    cluster_3_x.append(data[0])
    cluster_3_y.append(data[1])
  elif y_kmeans[index] == 3:
    cluster_4_x.append(data[0])
    cluster_4_y.append(data[1])


plt.figure(figsize=(15,7))
sns.scatterplot(cluster_1_x, cluster_1_y, color = 'yellow', label = 'Cluster 1')
sns.scatterplot(cluster_2_x, cluster_2_y, color = 'blue', label = 'Cluster 2')
sns.scatterplot(cluster_3_x, cluster_3_y, color = 'green', label = 'Cluster 3')
sns.scatterplot(cluster_4_x, cluster_4_y, color = 'red', label = 'Cluster 4')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'black', label = 'Centroids',s=100,marker=',')
plt.title('Clusters of Planets')
plt.xlabel('Planet Radius')
plt.ylabel('Planet Mass')
plt.legend()
plt.gca().invert_yaxis()
plt.show()