# -----------------------
# Set Up Paths & Modules
# -----------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Support_Files')))
parent_dir = os.path.dirname(os.getcwd())
views_dir = os.path.join(parent_dir, "Layer_Views")
sys.path.append(views_dir)
from support_functions import *

#----------------------------------------
#STEP 1 - Specify File Locations
#----------------------------------------
#TER3 is the original CRISM TER3 Image File
#h5_file is our processed CRISM TER3 Image File
#The mro_crism file is output of the processed MRO CRISM Library Samples

ter3_file = "D:/CRISM Data/FRT0000634B/TER_Data/frt0000634b_07_if163j_ter3.img"
h5_file = "D:/CRISM Data/FRT0000634B/TER_Data/Python_Converted/stacked_frames.h5"
mro_crism = parent_dir + '\\MRO_Spectra_Library_Conversion\\MRO_Library_Results.csv'

#Some outputs from the SOM model - Just usable pixel data extracted from the h5 file.
reshapped_data_loc = parent_dir + '\\SOM_Training\\Scans\\FRT0000634B_50_50\\reshaped_data.csv'
reshapped_locations_loc = parent_dir + '\\SOM_Training\\Scans\\FRT0000634B_50_50\\reshaped_indices.csv'

#----------------------------------------
#STEP 2 - KMeans Parameters
#----------------------------------------
number_clusters = 2500

#----------------------------------------
#STEP 4 - Mineral and Visual Preferences
#----------------------------------------
Mineral_Name = "Illite_Muscovite" #Use the index name in the mro_crism_library
Number_Clusters = 5

#Do you want to graph the correlation of each instance, clustered under each selected neuron?
graph_instance_correlation = False #True or False

#Do you want a Ground or Overlay Image View
Ground_View = True
bands = ["BD2165", "BD2250", "BD1900R2"] #RGB Bands for overlay image.

#Is there a target pixel to draw a cross?
target_pixel = False
target_location = (16, 330) #[X,Y] where X is vertical and Y the horizontal axis

#Do you want to mark clustered pixels?
mark_clustered_pixels = True
highlight_color = [255, 0, 0]

#Do you want to rotate the image (1 = 90 degrees, 2 = 180 degrees etc):
rotation = 2

#----------------------------------------
#Import Data
#----------------------------------------
#Import MRO CRISM Library
parent_dir = os.path.dirname(os.getcwd())
mro_crism_library = pd.read_csv(mro_crism, index_col=0)
clipped_mro = mro_crism_library.iloc[:, 2:]

#Import Data files
reshaped_pixels = np.genfromtxt(reshapped_data_loc, delimiter=",")
reshaped_locations = np.genfromtxt(reshapped_locations_loc, delimiter=",")

#----------------------------------------
#Instance Correlation to Mineral
#----------------------------------------
mineral_instance = clipped_mro.loc[Mineral_Name].values
correlation_array = np.empty(reshaped_pixels.shape[0])
for i in range(reshaped_pixels.shape[0]):
    row = reshaped_pixels[i]
    correlation_array[i] = np.corrcoef(row, mineral_instance)[0, 1]

# -------------------------------
# K-Means Clustering
# -------------------------------
kmeans = KMeans(n_clusters=number_clusters)
labels = kmeans.fit_predict(reshaped_pixels)

# Create a DataFrame for easier group by summarization
df = pd.DataFrame({
    'correlation': correlation_array,
    'cluster': labels
})

summary = df.groupby('cluster')['correlation'].describe()

# Sort by mean correlation in descending order
sorted_summary = summary.sort_values(by='mean', ascending=False)

#Find the cluster numbers
top_clusters = sorted_summary.index.tolist()[0:Number_Clusters]

#Find the instances belonging to those top 5 clusters
is_in_instance = np.isin(labels, top_clusters)

#Create reshaped pixel maps to fit in with the SOM version of this code
reshaped_pixels_maps = np.zeros((reshaped_pixels.shape[0], 3))
reshaped_pixels_maps[:, 1] = labels

#Create graph_nodes to fit with the SOM version of this code
graph_nodes = np.zeros((len(top_clusters), 4))
graph_nodes[:, 1] = np.array(top_clusters)

#Create closest_nodes to fit with the SOM version of this code
closest_nodes = np.zeros((sorted_summary.shape[0], 4))
closest_nodes[:,1] = np.array(sorted_summary.index.tolist())
closest_nodes[:,3] = np.array(sorted_summary.iloc[:,6].to_list())

#----------------------------------------
#Map correlation of each instance for that node
#----------------------------------------
if graph_instance_correlation:
    instance_correlation_graph(graph_nodes, mineral_instance, reshaped_pixels_maps, reshaped_pixels)

#----------------------------------------
#Extract the top "Number_Neurons" Neurons and identify pixels clusted by those Neurons.
#----------------------------------------
#reshaped_pixels_maps_match = reshaped_pixels_maps[is_in_instance]
reshaped_pixels_loc = reshaped_locations[is_in_instance]

#----------------------------------------
#Obtain Views
#----------------------------------------
if Ground_View:
    rgb_8bit_from_float = false_view(ter3_file)
else:
    rgb_8bit_from_float = call_layer(h5_file, bands)

#Draw cross on image
if target_pixel == True:
    rgb_8bit_from_float = draw_plus(rgb_8bit_from_float.copy(), target_location, size=50, thickness=2)

#Highlight clustered pixels
if mark_clustered_pixels:

    for loc in reshaped_pixels_loc.astype(int):  # Ensure pixel locations are integers
        row, col = loc
        rgb_8bit_from_float[row, col] = highlight_color


# On highlighted pixels, undertake DB Scan and morphological dilation to create a buffer around each groups.
points, point_indices, labels = db_scan_convex_hull(rgb_8bit_from_float, reshaped_pixels_loc, reshaped_locations)

# Loop through each grouped pixel, find the neuron it is mapped to and the correlation of that neuron to the mineral
point_correlation = []
point_node = []

for t_row in point_indices:

    if t_row == -1:
        point_correlation.extend([0])
        point_node.extend([0])
    else:
        tmp_node = reshaped_pixels_maps[t_row, 1]
        point_correlation.extend([closest_nodes[closest_nodes[:, 1] == tmp_node][0][3]])
        point_node.extend([tmp_node])

    #These results are used later to find the maximum correlation of the pixels clustered by each hull.

#Create a list to hold the legend labels and markers
legend_labels = []
legend_markers = []

# Draw convex hull around each cluster
#-------------------------------
unique_labels = set(labels)
colors = plt.cm.get_cmap('gist_rainbow', len(unique_labels))

#For each clustered region based on DB Scan, this finds the correlation ot each pixel
for i, label in enumerate(unique_labels):

    if label == -1:
        continue  # Skip noise points

    #Previously the neuron for each pixel within the convex hull was identifed and the correlation of that neuron
    #obtain, the following just extracts the correlation of the neuron which has the maximum correlation to the
    #mineral within that convex hull. Individual pixels may have an exceedingly high correlation, but the neuron it is
    #clustered under may not be so high.

    point_correlation = np.array(point_correlation)
    filtered_correlation = point_correlation[labels == int(label)]
    cluster_point_correlation = np.max(filtered_correlation)
    cluster_points = points[labels == label]  # Fix cluster points extraction

    # Apply a cut off for the minimum number of points to form a hull.
    if len(cluster_points) < 5:
        continue

    # Compute convex hull
    hull = cv2.convexHull(cluster_points.reshape(-1, 1, 2))
    hull_swapped = np.array([[[pt[0][1], pt[0][0]]] for pt in hull])

    # Ensure that the color is in the correct integer format
    hull_color = (np.array(colors(i)[:3]) * 255).astype(int)  # Convert normalized color to 0-255 range
    hull_color_tuple = tuple(hull_color)  # Convert to tuple

    # Get RGB values from colormap (scaled between 0 and 1)
    rgb_color_tuple = (colors(i)[2], colors(i)[1], colors(i)[0])  # RGB values

    # Convert the RGB tuple to BGR format and scale to [0, 255] for OpenCV
    bgr_color_tuple = (int(rgb_color_tuple[0] * 255), int(rgb_color_tuple[1] * 255), int(rgb_color_tuple[2] * 255))

    # Draw the hull with the correct color (in BGR format)
    cv2.polylines(rgb_8bit_from_float, [hull_swapped], isClosed=True, color=bgr_color_tuple, thickness=2)

    # Add a label and marker to the legend list with the same RGB color for Matplotlib
    legend_labels.append(f"Cluster {label} + 75th pctle corr {round(cluster_point_correlation, 3)}")
    legend_markers.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=rgb_color_tuple, markersize=10))


# Plot results (overlay hulls on rotated image)
rotated_image = np.rot90(rgb_8bit_from_float, k=rotation)
plt.figure(figsize=(10, 10))
plt.imshow(rotated_image)
plt.axis("off")
plt.legend(legend_markers, legend_labels, loc="upper right", fontsize=16)
plt.show()