# -----------------------
# Set Up Paths & Modules
# -----------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Support_Files')))
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
views_dir = os.path.join(parent_dir, "Layer_Views")
sys.path.append(views_dir)
from support_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------
#Specify File Locations
#----------------------------------------
#TER3 is the original CRISM TER3 Image File
#h5_file is our processed CRISM TER3 Image File
#The mro_crism file is output of the processed MRO CRISM Library Samples
ter3_file = "D:/CRISM Data/FRT00003E12/TER_Data/frt00003e12_07_if166j_ter3.img"
h5_file = "D:/CRISM Data/FRT00003E12/TER_Data/Python_Converted/stacked_frames.h5"
mro_crism = parent_dir + '\\MRO_Spectra_Library_Conversion\\MRO_Library_Results.csv'

#Outputs from the SOM's training
som_weights_loc = parent_dir + '\\SOM_Training\\Scans\\FRT00003E12_50_50\\som_weights.csv'
som_locations_loc = parent_dir + '\\SOM_Training\\Scans\\FRT00003E12_50_50\\som_locations.csv'
reshapped_data_loc = parent_dir + '\\SOM_Training\\Scans\\FRT00003E12_50_50\\reshaped_data.csv'
reshapped_locations_loc = parent_dir + '\\SOM_Training\\Scans\\FRT00003E12_50_50\\reshaped_indices.csv'

#----------------------------------------
#Identify Spatial Distribution of a Specific Mineral
#----------------------------------------
Mineral_Name = "Fe_Olivine" #Use the index name in the mro_crism_library
Number_Neurons = 5

#Do you wannt to graph the correlation of each instance, clustered under each selected neuron?
graph_instance_correlation = True #True or False

#Do you want a Ground or Overlay Image View
Ground_View = True
bands = ["OLINDEX3", "BD1300", "BD2290"] #RGB Bands for overlay image only.

#Is there a target pixel to draw a cross?
target_pixel = True
target_location = (401, 443) #[X,Y] where X is vertical and Y the horizontal axis

#Do you want to mark clustered pixels?
mark_clustered_pixels = True
highlight_color = [255, 0, 0]

#Do you want to rotate the image (1 = 90 degrees, 2 = 180 degrees etc):
rotation = 0

#Do you want to plot pixels under specific neurons
graph_select_nodes = False

neuron_list = np.array([  19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,
         30,   31,   32,   33,   34,   35,   36,   37,   44,   45,   67,
         68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,
         79,   80,   81,   82,   83,   84,   85,   86,   91,   92,   93,
         94,   95,   96,  118,  119,  120,  121,  122,  123,  124,  125,
        126,  127,  128,  129,  130,  131,  132,  133,  134,  135,  136,
        137,  142,  143,  144,  145,  146,  147,  168,  169,  170,  171,
        172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,
        183,  184,  185,  186,  187,  188,  191,  192,  193,  194,  195,
        196,  197,  198,  219,  220,  221,  222,  223,  224,  225,  226,
        227,  228,  229,  230,  231,  232,  233,  234,  235,  236,  237,
        238,  239,  240,  241,  242,  243,  244,  245,  246,  247,  248,
        269,  270,  271,  272,  273,  274,  275,  276,  277,  278,  279,
        280,  281,  282,  283,  284,  285,  286,  287,  289,  290,  291,
        292,  293,  294,  295,  296,  297,  298,  299,  320,  321,  322,
        323,  324,  325,  326,  327,  328,  329,  330,  331,  333,  334,
        335,  336,  337,  338,  339,  340,  341,  342,  343,  344,  345,
        346,  347,  348,  349,  370,  371,  372,  373,  374,  375,  376,
        377,  378,  379,  380,  381,  382,  383,  384,  385,  386,  387,
        388,  389,  390,  391,  392,  393,  394,  395,  396,  397,  398,
        399,  420,  421,  422,  423,  424,  425,  426,  427,  428,  429,
        430,  431,  432,  433,  434,  435,  436,  437,  438,  439,  440,
        441,  442,  443,  444,  445,  446,  447,  448,  449,  469,  470,
        471,  472,  473,  474,  475,  476,  477,  478,  479,  480,  481,
        482,  483,  484,  485,  486,  487,  488,  489,  490,  491,  492,
        493,  494,  495,  496,  497,  498,  499,  519,  521,  522,  523,
        524,  525,  526,  527,  528,  529,  530,  531,  532,  533,  534,
        535,  536,  537,  538,  539,  540,  541,  542,  543,  544,  545,
        546,  547,  548,  549,  569,  571,  572,  573,  574,  575,  576,
        577,  578,  579,  580,  581,  582,  583,  584,  585,  586,  587,
        588,  590,  591,  594,  596,  597,  598,  622,  623,  624,  625,
        626,  627,  628,  629,  630,  631,  632,  633,  634,  635,  636,
        637,  638,  639,  640,  641,  642,  647,  672,  673,  674,  675,
        676,  677,  678,  679,  680,  681,  682,  683,  684,  685,  686,
        687,  689,  690,  691,  692,  722,  723,  724,  725,  726,  727,
        728,  729,  730,  731,  732,  733,  734,  735,  736,  737,  738,
        739,  740,  741,  742,  748,  772,  773,  774,  775,  776,  777,
        778,  779,  780,  781,  782,  783,  784,  785,  786,  787,  788,
        789,  790,  791,  792,  798,  799,  826,  827,  828,  829,  830,
        831,  832,  833,  834,  835,  836,  837,  838,  839,  840,  841,
        842,  843,  878,  879,  880,  881,  882,  883,  884,  885,  886,
        887,  888,  889,  890,  891,  892,  893,  928,  929,  930,  931,
        932,  933,  934,  935,  936,  937,  938,  939,  940,  941,  942,
        944,  977,  978,  979,  980,  981,  982,  983,  984,  985,  986,
        987,  988,  989,  990,  991,  992,  993,  994,  995,  996,  997,
       1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036,
       1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047,
       1048, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084,
       1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095,
       1096, 1097, 1125, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134,
       1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145,
       1146, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184,
       1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195,
       1197, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235,
       1236, 1237, 1238, 1239, 1240, 1241, 1242, 1244, 1245, 1278, 1279,
       1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290,
       1296, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337,
       1338, 1340, 1346, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384,
       1385, 1386, 1387, 1388, 1427, 1428, 1429, 1430, 1431, 1432, 1433,
       1434, 1435, 1436, 1437, 1438, 1480, 1481, 1482, 1483, 1488, 1520,
       1530, 1531, 1532, 1533, 1535, 1537, 1539, 1569, 1570, 1571, 1574,
       1582, 1584, 1585, 1587, 1598, 1599, 1619, 1620, 1621, 1622, 1648,
       1670, 1683, 1722, 2349])

#----------------------------------------
#Import Data
#----------------------------------------
#Import MRO CRISM Library
parent_dir = os.path.dirname(os.getcwd())
mro_crism_library = pd.read_csv(mro_crism, index_col=0)
clipped_mro = mro_crism_library.iloc[:, 2:]

#Import SOM files
som_weights = np.genfromtxt(som_weights_loc, delimiter=",", skip_header=1)
som_locations = np.genfromtxt(som_locations_loc, delimiter=",", skip_header=1)

#Import Data files
reshaped_pixels = np.genfromtxt(reshapped_data_loc, delimiter=",")
reshaped_locations = np.genfromtxt(reshapped_locations_loc, delimiter=",")

#Cluster pixels according to the SOM Map
reshaped_pixels_maps = map_input_euclidean(som_weights, som_locations, reshaped_pixels)

#Order neurons by their match to the mineral input vector
mineral_instance = clipped_mro.loc[Mineral_Name].values
closest_nodes = map_input_euclidean_group(som_weights, som_locations, np.array(mineral_instance))

#----------------------------------------
#Extract the top "Number_Neurons" Neurons and identify pixels clusted by those Neurons.
#----------------------------------------

if graph_select_nodes == True:
    mask = np.isin(closest_nodes[:, 1], neuron_list)
    graph_nodes = closest_nodes[mask]
else:
    graph_nodes = closest_nodes[0:Number_Neurons, :]

reshaped_pixels_maps_match = np.isin(reshaped_pixels_maps[:, 1], graph_nodes[:, 1])
reshaped_pixels_loc = reshaped_locations[reshaped_pixels_maps_match]

#----------------------------------------
#Map correlation of each instance for that node
#----------------------------------------
if graph_instance_correlation:
    instance_correlation_graph(graph_nodes, mineral_instance, reshaped_pixels_maps, reshaped_pixels)

#----------------------------------------
#Obtain Views
#----------------------------------------
if Ground_View:
    rgb_8bit_from_float = false_view(ter3_file)
else:
    rgb_8bit_from_float = call_layer(h5_file, bands)

#Highlight clustered pixels
if mark_clustered_pixels:

    for loc in reshaped_pixels_loc.astype(int):  # Ensure pixel locations are integers
        row, col = loc
        rgb_8bit_from_float[row, col] = highlight_color

#Draw cross on image
if target_pixel == True:
    rgb_8bit_from_float = draw_plus(rgb_8bit_from_float.copy(), target_location, size=50, thickness=2, color = (255,255,0))

    #plt.figure(figsize=(10, 10))
    #plt.imshow(rgb_8bit_from_float)
    #plt.axis("off")
    #plt.show()

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
    legend_labels.append(f"Cluster {label} + max corr {round(cluster_point_correlation, 3)}")
    legend_markers.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=rgb_color_tuple, markersize=10))


# Plot results (overlay hulls on rotated image)
rotated_image = np.rot90(rgb_8bit_from_float, k=rotation)
plt.figure(figsize=(10, 10))
plt.imshow(rotated_image)
plt.axis("off")
plt.legend(legend_markers, legend_labels, loc="upper right", fontsize=16)
plt.show()