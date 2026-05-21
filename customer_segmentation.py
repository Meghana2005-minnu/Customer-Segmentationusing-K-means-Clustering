# ============================================
# Customer Segmentation using K-Means
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================
# Step 1: Load Dataset
# ============================================

data = pd.read_csv("dataset/Mall_Customers.csv")

print("\nFirst 5 Rows:\n")
print(data.head())

print("\nDataset Information:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

# ============================================
# Step 2: Data Preprocessing
# ============================================

# Encode Gender Column
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])

# Select Features
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# Step 3: Find Optimal Number of Clusters
# Using Elbow Method
# ============================================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
    n_clusters=i,
    init='k-means++',
    random_state=42,
    n_init=10
)

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.savefig("outputs/elbow_method.png")
plt.show()

# ============================================
# Step 4: Apply K-Means Clustering
# ============================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_kmeans = kmeans.fit_predict(X_scaled)

# Add Cluster Labels
data['Cluster'] = y_kmeans

print("\nClustered Data:\n")
print(data.head())

# ============================================
# Step 5: Visualization
# ============================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='Set1',
    s=100
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.savefig("outputs/customer_clusters.png")
plt.show()

# ============================================
# Step 6: Cluster Analysis
# ============================================

cluster_summary = data.groupby('Cluster').mean()

print("\nCluster Summary:\n")
print(cluster_summary)

# ============================================
# Step 7: Business Insights
# ============================================

print("\nBusiness Insights:\n")

for cluster in sorted(data['Cluster'].unique()):
    print(f"Cluster {cluster}:")
    
    avg_income = data[data['Cluster'] == cluster]['Annual Income (k$)'].mean()
    avg_score = data[data['Cluster'] == cluster]['Spending Score (1-100)'].mean()

    if avg_income > 70 and avg_score > 60:
        print("  -> High income and high spending customers")
    
    elif avg_income < 40 and avg_score < 40:
        print("  -> Low income and low spending customers")
    
    else:
        print("  -> Moderate customers")

print("\nProject Completed Successfully!")