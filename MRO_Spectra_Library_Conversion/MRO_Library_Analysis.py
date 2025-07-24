import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Import Processed MRO Library Data & Clip
mro_data = pd.read_csv("MRO_Library_Results.csv", index_col=0)
mro_data_clipped = mro_data.iloc[:,2:]

#Extract Iron Oxides & Silicates
#---------------------------
mro_sulfate_data = mro_data[mro_data["Mineral Group"] == "Iron oxides and primary silicates"]
mro_FE_names = mro_sulfate_data.iloc[:,0].tolist()
mro_FE_first_ten = mro_sulfate_data.iloc[:,2:12]

# Create heatmap
plt.figure(figsize=(16, 4))
ax = sns.heatmap(mro_FE_first_ten, annot=True, cmap="coolwarm", fmt=".2f", yticklabels=mro_FE_names, linewidths=0.5, linecolor='white')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel("First 10 Summary Products", fontsize=12, labelpad=10)
ax.set_ylabel("Iron Oxides & Silicates", fontsize=12)
plt.xticks(rotation=0)
plt.savefig("heatmap_output.png", dpi=300, bbox_inches="tight")

#Correlation Map for all mineral
#---------------------------
correlation_matrix = mro_data_clipped.T.corr()
correlation_matrix = correlation_matrix.iloc[1:, :-1]
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
plt.figure(figsize=(18, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, linecolor='none', mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Mineral Correlation Heatmap", fontsize=16)
plt.xlabel("Minerals", fontsize=14)
plt.ylabel("Minerals", fontsize=14)
plt.savefig("correlation_map_minerals.png", dpi=300, bbox_inches="tight")

#Clip data is if above 30th percentile it gets 1 or 0
mro_scaled = mro_data_clipped.copy()
for col in mro_scaled.columns:

    # Set values ≤ 0 to 0
    mro_scaled[col] = np.where(mro_scaled[col] <= 0, 0, mro_scaled[col])
    threshold = np.percentile(mro_scaled[col], 30)
    mro_scaled[col] = np.where((mro_scaled[col] > threshold) & (mro_scaled[col] > 0), 1, 0)

correlation_matrix = mro_scaled.T.corr()
correlation_matrix = correlation_matrix.iloc[1:, :-1]
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
plt.figure(figsize=(16, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, linecolor='none', mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Mineral Correlation Heatmap", fontsize=16)
plt.xlabel("Minerals", fontsize=12)
plt.ylabel("Minerals", fontsize=12)
plt.show()
plt.savefig("correlation_map_scaled_minerals.png", dpi=300, bbox_inches="tight")
