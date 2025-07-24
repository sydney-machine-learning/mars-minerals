import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter

# -----------------------
# Set Up Paths
# -----------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Support_Files')))
from support_functions import * #Located in the Support_Files directory
parent_dir = os.path.dirname(os.getcwd())

# -----------------------
# Specify SOM Output Locations
# -----------------------
som_weights_loc = "https://zenodo.org/records/16397494/files/som_weights.csv"
som_locations_loc = "https://zenodo.org/records/16397494/files/som_locations.csv"
reshapped_data_loc = "https://zenodo.org/records/16397494/files/reshaped_data.csv"
reshapped_locations_loc = "https://zenodo.org/records/16397494/files/reshaped_indices.csv"

#Import MRO CRISM Library
parent_dir = os.path.dirname(os.getcwd())
mro_crism_library = pd.read_csv(f"{parent_dir}/MRO_Spectra_Library_Conversion/MRO_Library_Results.csv", index_col=0)
clipped_mro = mro_crism_library.iloc[:, 2:]

#Import SOM files
som_weights = np.genfromtxt(som_weights_loc, delimiter=",", skip_header=1)
som_locations = np.genfromtxt(som_locations_loc, delimiter=",", skip_header=1)
reshapped_data = np.genfromtxt(reshapped_data_loc, delimiter=",", skip_header=1)

#Label Neurons with a Mineral
mineral_names = mro_crism_library.loc[:,"Mineral Name"].tolist()
mineral_3D = neuron_mineral_grouping(clipped_mro, som_weights, som_locations, mineral_names)

# Plot the number of minerals mapped to each neuron
pixel_sums = np.sum(mineral_3D, axis=0)
pixel_sums_flatten = pixel_sums.flatten()
plot_heatmap(pixel_sums_flatten, som_locations, "Number of Minerals", "Neuron Mineral Mapping")

#Find what minerals are mapped to each corner
Top_Left =[]
Top_Right =[]
Bottom_Left = []
Bottom_Right = []

Full_List = ['Carbonates', 'Phyllosilicates', 'Iron oxides and primary silicates', 'Ices',
             'Other hydrated silicates and halides', 'Sulfates']

Short_List = ['Carbs', 'Phyllos', 'FE Silicates', 'Ices', 'Hydr. sil & hal', 'Sulfates']

rows = mineral_3D.shape[1]
cols = mineral_3D.shape[2]

for r in range(rows):
    for c in range (cols):
        for m in range (mineral_3D.shape[0]):

            #is the mineral located in that pixel
            if mineral_3D[m,r,c] == 1:
                mineral_group = mro_crism_library.loc[mro_crism_library.index[m], "Mineral Group"]


                #Swap to short version of the name
                m_index = Full_List.index(mineral_group)
                mineral_group = Short_List[m_index]

                # Classify based on corner location
                if r < rows // 3 and c < cols // 3:
                    Top_Left.append(mineral_group)
                elif r < rows // 3 and c >= (cols - (cols// 3)):
                    Top_Right.append(mineral_group)
                elif r >= ((rows - rows // 3)) and c < cols // 3:
                    Bottom_Left.append(mineral_group)
                elif r >= ((rows - rows // 3)) and c >= (cols - (cols// 3)):
                    Bottom_Right.append(mineral_group)
                else:
                    continue
#Normalise Count
corner_data = {}

# Process each corner
for corner in ["Top_Left", "Top_Right", "Bottom_Left", "Bottom_Right"]:
    minerals_list = eval(corner)
    mineral_counts = Counter(minerals_list)
    total_count = sum(mineral_counts.values())

    # Normalize counts
    normalized_counts = {mineral: count / total_count for mineral, count in mineral_counts.items()}
    corner_data[corner] = normalized_counts

# Get all unique minerals across all corners (union of all keys)
all_minerals = list(set(mineral for data in corner_data.values() for mineral in data.keys()))

# Create mapping for mineral groups
mineral_group_map = {mineral: f"Group {i+1}" for i, mineral in enumerate(all_minerals)}

# Process each corner
for corner in ["Top_Left", "Top_Right", "Bottom_Left", "Bottom_Right"]:
    minerals_list = eval(corner)
    mineral_counts = Counter(minerals_list)
    total_count = sum(mineral_counts.values())

    # Normalize counts
    normalized_counts = {mineral: count / total_count for mineral, count in mineral_counts.items()}
    corner_data[corner] = normalized_counts

# Get all unique minerals across all corners (union of all keys)
all_minerals = list(set(mineral for data in corner_data.values() for mineral in data.keys()))

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
corner_titles = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
main_title = "Distribution of Mineral Mapping Relative to SOM Corners"
fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.96)

# Create the plots
for ax, (corner, data) in zip(axes.flatten(), corner_data.items()):
    # Ensure all minerals are in the same order and zeroed if not present
    values = [data.get(mineral, 0) for mineral in all_minerals]

    ax.bar(all_minerals, values, color='skyblue')
    ax.set_title(f"{corner.replace('_', ' ')}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Normalized Frequency", fontsize=12)
    ax.set_xticklabels(all_minerals, rotation=35, ha="right", fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    ax.tick_params(axis='y', labelsize=12)

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#Find the hit rate for each Neuron
hits = find_bmu_group(som_weights, reshapped_data)
neuron_hits = list(range(0, 2500))
for n in neuron_hits:
    neuron_hits[n] = int(np.sum(hits==n))

plot_heatmap(neuron_hits, som_locations, "Number of Hits", "Neuron Hits Heat Map")

#U-Matrix
u_matrix = compute_u_matrix(som_weights, som_locations)
plot_heatmap(u_matrix, som_locations, "Distance", "SOM U-Matrix")
