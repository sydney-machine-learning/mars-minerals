#This script trains a SOM model based on the instances contained in the h5 file.
#h5 file required, generated in CRISM_Spectral_Conversion.py

# -----------------------
# STEP 1/3 - Datasource
# -----------------------
file_path = "D:/CRISM Data/FRT00003E12/Python_Converted/stacked_frames.h5"

# -----------------------
# STEP 2/3 SOM Training Parameters
# -----------------------
X = 50                 # SOM Size (X)
Y = 50                 # SOM Size (Y)
samples_iter = 5000    # Number of instances per iteration
l_rate = 0.5           # Initial learning rate
burn_itter = 50        # Burn-in iterations
num_itter = 1000       # Learning iterations

# -----------------------
# STEP 3/3 - Minimum Group Size
# -----------------------
min_instance = 2000  # Number of instances for smallest group.

#Kmeans is used to segregate data into groups until the size of the smallest group contains less than the number of
#instances specified here, instances are then sampled equally from each of these groups to ensure smaller groups, which
#are more likely to contain interesting features obtain more representation in the SOM training.

# -----------------------

#***************************************

# -----------------------
#Import Statements
# -----------------------
import os
import sys
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from som_class import *

# -----------------------
# Set Up Paths
# -----------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Support_Files')))
from support_functions import *

# -----------------------
# Import Processed TER3 Data (H5 File)
# -----------------------
with h5py.File(file_path, "r") as f:
    combined_data = f['stacked_frames'][:]  # Load all data at once

# -------------------------------
# Preprocessing - Scaling
# -------------------------------

# Apply scaling to all frames
data_lists = [preprocess_data(frame) for frame in combined_data]
combined_data = np.stack(data_lists, axis=-1)

# -------------------------------
# Create Valid Input Data
# -------------------------------
valid_mask = np.all(combined_data != 65535, axis=-1)  # Mask pixels with any 65535
valid_indices = np.argwhere(valid_mask)

reshaped_data = combined_data[valid_mask].reshape(len(valid_indices), -1)
reshaped_data = np.array(reshaped_data, dtype=np.float32)

del combined_data, data_lists  # Free memory

# -------------------------------
# K-Means Clustering
# -------------------------------
cluster_assignments = np.full(reshaped_data.shape[0], -1, dtype=int)
min_count = min_instance * 2
k = 1

while min_count > min_instance:
    kmeans = KMeans(n_clusters=k).fit(reshaped_data)
    labels = kmeans.fit_predict(reshaped_data)
    _, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    k += 1

cluster_assignments[:] = kmeans.labels_

# -------------------------------
# Train SOM
# -------------------------------
start_radius = max(X, Y) / 2
som = SOM(x=X, y=Y, input_dim=reshaped_data.shape[1],
          learning_rate=l_rate, num_iter=num_itter, radius=start_radius)

som.train(reshaped_data, cluster_assignments, samples_iter, burn_itter)

# Export SOM and frame data
som.export_to_csv()
np.savetxt("reshaped_indices.csv", valid_indices, delimiter=',', fmt='%.9f')
np.savetxt("reshaped_data.csv", reshaped_data, delimiter=',', fmt='%.9f')

# -------------------------------
# Post-Training: SOM Groupings & Visualization
# -------------------------------
mapped_centroids = som.map_input(reshaped_data)
mid_matrix = som.compute_mid()

plt.imshow(mid_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Mean Inter-neuron Distances (MID) Heatmap")
plt.show()