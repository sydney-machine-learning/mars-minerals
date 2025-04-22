import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import cv2
import h5py
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr

class Data_Package:
    def __init__(self, spectrum_data):

        #Store Spectral data as a stable
        self.spectral_table = spectrum_data

        #Create dictionary of the same
        self.spectral_dict = {}
        for index, tmp_row in self.spectral_table.iterrows():
            tmp_name = f"Band_{index+1}"
            self.spectral_dict[tmp_name] = f"{tmp_row['Wavelength']} Nanometers"

    def tags(self):
        return self.spectral_dict

    def read(self, wavelength):

        #Return a wavelength the same as a Rasterio object
        return self.spectral_table.loc[wavelength-1,"Numerator"]

    def __repr__(self):
        return f"Dataset({self.spectrum})"

def clean_data(web_link):

    # Import Mg_Carbonate Signature
    df = pd.read_csv(web_link, delimiter=", ", engine="python")

    # Extract Data
    Wavelength = df.iloc[:, 0] * 1000
    Numerator = df.iloc[:, 4]

    # Remove no data readings (value of 65535)
    bad_data = Numerator[Numerator == 65535].index
    Wavelength = Wavelength.drop(bad_data).reset_index(drop=True)
    Numerator = Numerator.drop(bad_data).reset_index(drop=True)

    # Check for extreme outliers
    median_IF = np.median(Numerator)
    max_IF = np.max(Numerator)

    if (max_IF / median_IF) > 5:
        bad_data = Numerator[Numerator < median_IF * 5].index
        Wavelength = Wavelength.drop(bad_data).reset_index(drop=True)
        Numerator = Numerator.drop(bad_data).reset_index(drop=True)

    df = pd.DataFrame({"Wavelength": np.round(Wavelength,5), "Numerator": Numerator})

    return df.sort_values(by="Wavelength").reset_index(drop=True)

def kernal_width(length, wavelengths, kernel, dataset):

    """
    Computes the reflectance at a specific wavelength for multispectral and hyperspectral data.
    - For hyperspectral, it uses the kernel width (e.g., R2230: 5) to median the closest bands.
    - For multispectral, it directly uses the available band at the specified wavelength.
    """
    # Case when kernel is 0 (no median calculation, just use the specific wavelength)
    if kernel == 0:
        closest_index = np.argmin(np.abs(wavelengths - length))
        array_list = [dataset.read(int(closest_index) + 1)]
        return [array_list[0], wavelengths[closest_index]]

    # Case for hyperspectral data (kernel width is provided)

    # Find the indices of the closest wavelengths
    distances = np.abs(wavelengths - length)
    closest_indices = np.argsort(distances)[:kernel]

    # Read the reflectance values for the closest bands
    array_list = [dataset.read(int(idx) + 1) for idx in closest_indices]

    # Compute the median reflectance over the selected bands
    stacked_arrays = np.stack(array_list, axis=0)
    median_2d = np.median(stacked_arrays, axis=0)

    return [median_2d, wavelengths[closest_indices]]

def CRISM_product_list():

    Product_List = ["OLINDEX3", "BD1300", "LCPINDEX2", "HCPINDEX2", "BD1400", "BD1435", "BD1500_2", "BD1750_2",
                    "BD1900_2", "BD1900R2", "BD2100_2", "BD2165", "BD2190", "MIN2200", "BD2210_2", "D2200", "BD2230",
                    "BD2250", "MIN2250", "BD2265", "BD2290", "D2300", "BD2355", "SINDEX2", "MIN2295_2480",
                    "MIN2345_2537", "BD2500_2", "CINDEX2", "R3920"]

    return Product_List

def normalize(band, view_name):

    #Mask 65535 values
    mask = band == 65535

    if view_name != "False_Color":

        lower_limit = np.percentile(band[~mask], 0.1)
        upper_limit = np.percentile(band[~mask], 99.9)

        # Apply np.clip only to the non-NaN values
        #band[~mask] = np.clip(band[~mask], lower_limit, upper_limit)
        band[~mask] = np.where(band[~mask] <= lower_limit, 0, band[~mask])
        band[~mask] = np.where(band[~mask] >= upper_limit, upper_limit, band[~mask])

        #Make sure nothing in the lower_limit is less than 0
        band[~mask] = np.where(band[~mask] <= 0, 0, band[~mask])

    #Remove out of range data
    band[mask] = np.nan

    #Normalise band
    band_min = np.nanmin(band)
    band_max = np.nanmax(band)
    normalised_band = (band - band_min) / (band_max - band_min)

    return np.nan_to_num(normalised_band, nan=0)

def draw_plus(image, center, size=50, thickness=10, color = (255,255,0)):

    # Function to draw a yellow plus symbol on an image
    y, x = center

    # Draw the horizontal red line (center of the cross)
    cv2.line(image, (x - size // 2, y), (x + size // 2, y), color, thickness, lineType=cv2.LINE_8)  # Red horizontal

    # Draw the vertical red line (center of the cross)
    cv2.line(image, (x, y - size // 2), (x, y + size // 2), color, thickness, lineType=cv2.LINE_8)  # Red vertical

    return image

def map_input_euclidean(som_weights, som_locations, input_vector):

    """
    This function finds which neuron is the closest match to each input_vector, this differs to the
    map_input_euclidean_group function which finds how well each neuron matches to an input vector.
    """

    #Store results - Closest Node, Euclidean Error and Correlation
    results_array = np.empty([input_vector.shape[0],3], dtype=object)

    i = 0
    for tmp_row in input_vector:

        #Euclidean Distance
        distances = np.linalg.norm(som_weights - tmp_row, axis=1)
        closest_index = np.argmin(distances)
        corr_matrix = np.corrcoef(som_weights[closest_index], tmp_row)

        results_array[i,0] = som_locations[closest_index]
        results_array[i, 1] = closest_index
        results_array[i, 2] = corr_matrix[1,0]

        i += 1

    return results_array

def map_input_euclidean_group(som_weights, som_locations, input_vector):

    """
    This function finds how close each neuron is to an input_vector, this differs to the map_input_euclidean function
    which only identifies which neuron is the closest match to each input_vector.
    """

    # Store results - Closest Node Locations, Closest Indices, and Correlation Values
    results_array = np.empty((som_weights.shape[0], 4), dtype=object)

    row_idx = 0  # Row index for the results_array

    # Loop through each node and obtain stats
    for r_index, r_row in enumerate(som_weights):

        # Compute correlation between the input vector and the closest node
        corr_matrix = np.corrcoef(r_row, input_vector)
        correlation = corr_matrix[0, 1]  # Extract correlation coefficient

        # Store the results: Locations, Indices, and Correlation values
        results_array[row_idx, 0] = som_locations[row_idx]
        results_array[row_idx, 1] = row_idx
        results_array[row_idx, 2] = np.linalg.norm(r_row - input_vector)
        results_array[row_idx, 3] = correlation

        row_idx += 1  # Move to the next row in the results array


    # Sory array based on correlation
    results_array = results_array[results_array[:, 3].argsort()[::-1]]

    return results_array

def instance_correlation_graph(graph_nodes, mineral_instance, reshaped_pixels_maps, reshaped_pixels):
    """
    This function plots the correlation of each instance to the neuron it is clusted under.

    Inputs are:
    - graph_nodes - A list of neurons to extract instances for, the first column [1] must contain the neuron number.
    - mineral_instance - A 1D ndarray with the weights of the mineral, corresponding to the weights of the SOM model.
    - reshaped_pixels_maps - An ndarray in which the first column [1] specifies the neuron that instance is clustered under.
    - reshaped_pixels - The weights for each instance
    """

    # Setup Plot
    colors = plt.cm.get_cmap('gist_rainbow', graph_nodes.shape[0])
    plt.figure(figsize=(10, 10))

    # Store plot ranges
    min_x = 1
    max_x = 0

    # Loop through each neuron and find the instances clustered by that neuron and the correlation to that neuron
    for tmp_r in range(graph_nodes.shape[0]):

        # Neuron Number
        neuron_number = graph_nodes[tmp_r, 1]
        mineral_node_match = reshaped_pixels_maps[:, 1] == neuron_number
        mineral_instances = reshaped_pixels[mineral_node_match]

        # Store and find the correlation of each instance
        correlations = []
        for instance in mineral_instances:
            corr = np.corrcoef(instance, mineral_instance)[0, 1]
            correlations.append(corr)

        # If there's data, plot it
        correlations_array = np.array(correlations)
        if correlations_array.size > 0:
            plt.hist(
                correlations_array,
                bins=30,
                alpha=0.6,
                label=f"Neuron {int(neuron_number)}",
                color=colors(tmp_r)
            )

        # Get temporary graph ranges
        tmp_min_x = max(0, np.floor(np.min(correlations) * 10) / 10 - 0.1)
        tmp_max_x = min(np.ceil(np.max(correlations) * 10) / 10 + 0.1, 1)
        min_x = min(tmp_min_x, min_x)
        max_x = max(tmp_max_x, max_x)

    # Add labels and title
    plt.xlim(min_x, max_x)
    plt.xlabel("Correlation Values", fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    plt.title("Distribution of Instance Correlation", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Add legend
    plt.legend(fontsize=18)  # Show labels for each neuron

    # Show the graph
    plt.show()

def false_view(ter3_location):

    """
    :param  This function requires a raw TER3 inage file, it will extract the spectral frequency closest to 2529,
            1506 and 1080 to create a false view of the ground.
    :return: RGB image.
    """

    with rasterio.open(ter3_location) as dataset:

        tags = dataset.tags()
        frequencies = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

        wave2529 = np.argmin(np.abs(frequencies - 2529))
        wave1506 = np.argmin(np.abs(frequencies - 1506))
        wave1080 = np.argmin(np.abs(frequencies - 1080))

        #Target Bands
        band_nir = dataset.read(int(wave2529)+1)
        band_red = dataset.read(int(wave1506) + 1)
        band_green = dataset.read(int(wave1080) + 1)

        nir_norm = normalize(band_nir, "False_Color")
        red_norm = normalize(band_red, "False_Color")
        green_norm = normalize(band_green, "False_Color")

        # Stack into an RGB false-color image
        false_color_img = np.dstack((nir_norm, red_norm, green_norm))

        # Convert to 8-bit by scaling and converting to uint8
        rgb_8bit_from_float = (false_color_img * 255).astype(np.uint8)

        # Rotate the image 270 degrees counterclockwise (or 90 degrees clockwise)
        #rgb_8bit_from_float = np.rot90(rgb_8bit_from_float, k=2)  # k=3 means 3 * 90 degrees

        # Plot the rotated image
        #plt.figure(figsize=(10, 10))
        #plt.imshow(rgb_8bit_from_float)
        #plt.axis("off")
        #plt.show()

    return rgb_8bit_from_float

def import_h5_file(file_location, band_names):
    """
    - Imports the processed TER3 data and extracts 3 bands required for an RGB view.
    - Band ordering is red, green and blue.
    """

    # Import processed TER3 files
    with h5py.File(file_location, "r") as f:
        combined_data = f['stacked_frames'][:]  # Load all data at once

    # Band ordering
    bands = CRISM_product_list()

    # Target Bands
    band_nir = combined_data[bands.index(band_names[0])]
    band_red = combined_data[bands.index(band_names[1])]
    band_green = combined_data[bands.index(band_names[2])]

    # Stack into an RGB false-color image
    img_stack = np.dstack((band_nir, band_red, band_green))

    return img_stack

def normalize_bands(img_stack, view_name, band_names):
    # Mask 65535 values
    mask = np.any(img_stack == 65535, axis=2)

    stretch_products = ["R530", "R440", "R600", "R770", "R1080", "R1506", "R2529", "R3920", "SH600_2", "IRA",
                        "ISLOPE1", "IRR2"]

    # Loop through the layers
    for i in range(img_stack.shape[2]):

        band = img_stack[:, :, i]
        band_name = band_names[i]

        if view_name != "False_Color":

            if band_name in stretch_products:
                lower_limit = np.percentile(band[~mask], 0.1)
                upper_limit = np.percentile(band[~mask], 99.9)

            else:
                lower_limit = 0
                upper_limit = np.percentile(band[~mask], 99.9)

            # Apply np.clip only to the non-NaN values
            band[~mask] = np.where(band[~mask] <= lower_limit, 0, band[~mask])
            band[~mask] = np.where(band[~mask] >= upper_limit, upper_limit, band[~mask])

            # Make sure nothing in the lower_limit is less than 0
            band[~mask] = np.where(band[~mask] <= 0, 0, band[~mask])

        # Remove out of range data
        band[mask] = np.nan

        # Normalise band
        band_min = np.nanmin(band)
        band_max = np.nanmax(band)
        img_stack[:, :, i] = (band - band_min) / (band_max - band_min)

    return np.nan_to_num(img_stack, nan=0)

def call_layer(file_location, band_names):

    img_stack = import_h5_file(file_location, band_names)

    norm_img_stack = normalize_bands(img_stack, "bland", band_names)

    # Convert to 8-bit by scaling and converting to uint8
    rgb_8bit_from_float = (norm_img_stack * 255).astype(np.uint8)

    rgb_8bit_from_float = np.rot90(rgb_8bit_from_float, k=0)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_8bit_from_float)
    plt.axis("off")
    plt.show()

    return rgb_8bit_from_float

def db_scan_convex_hull(rgb_8bit_from_float, reshaped_pixels_loc, reshaped_locations):

    """
    :param rgb_8bit_from_float: RGB Image file, just used for dimensions.
    :param reshaped_pixels_loc: Pixels that are clustered to the selected neurons.
    :param reshaped_locations: The location of all pixels.
    :return:
        - point_indices: The indices of each pixel within the buffer region.
        - labels: The DB Scan label for each pixel.
    """

    # Create binary mask for marked pixels
    mask = np.zeros_like(rgb_8bit_from_float[:, :, 0], dtype=np.uint8)
    mask[reshaped_pixels_loc[:, 0].astype(int), reshaped_pixels_loc[:, 1].astype(int)] = 255

    # Step 1: Extract coordinates of marked pixels in the mask
    points = np.column_stack(np.where(mask > 0))

    # Step 2: Run DBSCAN to cluster nearby points
    dbscan = DBSCAN(eps=35, min_samples=2)
    labels = dbscan.fit_predict(points)

    # Step 3: Remove points labeled as -1 (outliers)
    filtered_points = points[labels != -1]

    # Step 4: Create a new mask with the filtered points
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    for pt in filtered_points:
        filtered_mask[pt[0], pt[1]] = 255

    # Step 5: Apply morphological dilation to the updated mask to create a buffer
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=2)

    # Step 6: Cluster the morphological dilation mask
    points = np.column_stack(np.where(dilated_mask > 0))
    dbscan = DBSCAN(eps=25, min_samples=10)
    labels = dbscan.fit_predict(points)

    # This is all about tracing back each pixel selected to the correlation of the neuron it belongs to
    # -------------------------------
    point_indices = []

    # Find the row location of each point
    for t_row in points:
        tmp_list = np.where((reshaped_locations[:, 0] == t_row[0]) & (reshaped_locations[:, 1] == t_row[1]))[0]
        if len(tmp_list) > 0:
            point_indices.append(tmp_list.item())
        else:
            point_indices.append(-1)

    return [points, point_indices, labels]

def preprocess_data(data):
    """
    Scale CRISM TER3 data while ignoring 65535 values.
    Inputs:
        data - A single frame for a particular summary product.
    Returns:
        Scaled data with masked invalid entries.
    """
    masked_data = np.ma.masked_equal(data, 65535)
    masked_data[masked_data < 0] = 0  # Clip negative values

    data_min = masked_data.min()  # Should be 0
    data_max = masked_data.max()
    scaled_data = (masked_data - data_min) / (data_max - data_min)

    data[~masked_data.mask] = scaled_data.compressed()
    return data

def find_bmu(som_weights, input_vect):

    """
    :param som_weights: The weights for each neuron.
    :param input_vect: The mineral vector.
    :return: The best matching neuron for a given input vector.
    """
    distances = np.linalg.norm(som_weights - input_vect, axis=1)
    return np.argmin(distances)

def find_bmu_group(som_weights, input_vectors):

    """
    :param som_weights: The weights for each neuron.
    :param input_vectors: A group of mineral vectors.
    :return: Array of BMU's for the mineral vectors.
    """

    # Initialize bmu_array with the number of rows in input_vectors
    bmu_array = np.zeros(input_vectors.shape[0], dtype=int)

    tmp_index = 0

    # Iterate over each row (vector) in input_vectors
    for row in input_vectors:
        tmp_bmu = find_bmu(som_weights, row)
        bmu_array[tmp_index] = tmp_bmu
        tmp_index += 1

    return bmu_array

def compute_u_matrix(weights, locations):
    """Compute U-Matrix based on neuron locations and weight distances."""
    num_neurons = len(weights)
    u_matrix = np.zeros(num_neurons)

    for i in range(num_neurons):
        x_i, y_i = locations[i]
        neighbors = []

        # Find neighboring neurons (manhattan distance ≤ 1)
        for j in range(num_neurons):
            x_j, y_j = locations[j]
            if abs(x_i - x_j) + abs(y_i - y_j) == 1:
                neighbors.append(weights[j])

        # Compute average distance to neighbors
        if neighbors:
            u_matrix[i] = np.mean([np.linalg.norm(weights[i] - n) for n in neighbors])

    return u_matrix

def plot_heatmap(matrix, locations, c_label, g_title):

    x_coords = locations[:, 0]
    y_coords = locations[:, 1]

    # Reshape based on the grid dimensions
    x_dim = max(x_coords) + 1
    y_dim = max(y_coords) + 1
    matrix_grid = np.zeros((int(y_dim), int(x_dim)))

    for i, (x, y) in enumerate(locations):
        matrix_grid[int(x),int(y)] = matrix[i]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix_grid, cmap="coolwarm", origin="upper")

    # Attach colorbar to the figure
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(c_label, fontsize=22)

    # Set title
    ax.set_title(g_title, fontsize=22)

    ax.tick_params(axis='both', labelsize=16)

    ax.xaxis.set_ticks_position('top')

    plt.show()

def neuron_mineral_grouping(clipped_mro, som_weights, som_locations, mineral_names):

    """
    :param clipped_mro: Contains the scaled weights for each known mineral
    :param som_weights: Weights for each Neuron in the grid.
    :param som_locations: Location of each Neuron in the grid.
    :param mineral_names: A list of mineral names in the order they are looped through.

    :return:    This function returns a 3D array with 1 indicating the mineral is mapped to that Neuron.
                Many minerals can be mapped to a Neuron.
    """

    # Compute correlation between each mineral and all rows in som_weights
    correlation_matrix = np.zeros((clipped_mro.shape[0], som_weights.shape[0]))
    for i in range(clipped_mro.shape[0]):
        for j in range(som_weights.shape[0]):
            correlation_matrix[i, j] = pearsonr(clipped_mro.iloc[i], som_weights[j])[0]

    # For grouping, determine a high and ordinary correlation
    high_correlation = np.percentile(correlation_matrix, 95)
    low_correlation = np.percentile(correlation_matrix, 50)

    # Create a 3D array to store results for all minerals
    mineral_3D = np.zeros(
        (clipped_mro.shape[0], int(np.max(som_locations[:, 0]) + 1), int(np.max(som_locations[:, 1]) + 1)), dtype=int)

    # Loop through the neuron correlation for each mineral and cluster
    for index in range(correlation_matrix.shape[0]):

        #Extract correlation and calculate z-score
        tmp_cor = correlation_matrix[index, :]
        z_scores = (tmp_cor - np.mean(tmp_cor)) / np.std(tmp_cor)

        # Step 1: Threshold high z-scores (e.g., z > 2)

        mask = tmp_cor[tmp_cor > high_correlation]
        if mask.shape[0]:
            #If a mask has high correlation, don't loose it, accept a more average z-scores
            threshold = np.minimum(2, np.mean(z_scores[tmp_cor > high_correlation]))
        else:
            #Otherwise only accept pixels with a high z-score signalling distinct uniqueness
            threshold = 2

        plot_heatmap(tmp_cor, som_locations, "Correlation", mineral_names[index] + " Neuron Correlation")

        #Find the neurones that this mineral exceeds the z_score threshold whilst acheiving the minimum correlation
        #requirement
        high_z_mask = ((z_scores >= threshold) & (tmp_cor >= low_correlation)).astype(int)
        plot_heatmap(high_z_mask, som_locations, "Identified Region", "Neurons Mapped to " + mineral_names[index])

        #Save the results to the 3D array
        for i, (x, y) in enumerate(som_locations):
            mineral_3D[index, int(x), int(y)] = high_z_mask[i]

    return mineral_3D
