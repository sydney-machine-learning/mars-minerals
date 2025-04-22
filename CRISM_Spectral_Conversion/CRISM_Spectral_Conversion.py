#This script consumes TER3 CRISM spectral data and converts that data into 29 summary products.
#The formulae for the 29 summary products are contained within the support_functions.py file.
#The output is stored as a h5py file.
#TER Data - https://ode.rsl.wustl.edu/mars/pagehelp/Content/Missions_Instruments/Mars%20Reconnaissance%20Orbiter/CRISM/CRISM%20Product%20Primer/CRISM%20TER.htm
#This script requires Version 3 TER data, there should be a .hdr, .img, .lbl file in the same directory.
#This script directly references to .img file.

#STEP 1 of 1 - SPECIFY INPUT AND OUTPUT LOCATIONS
#---------------------------------
#TER3 is the raw TER3 image file, in the same folder should be a .hdr and .lbl file.
TER3_Location = "D:/CRISM Data/FRT00003E12/IF166J_TER/frt00003e12_07_if166j_ter3.img"
Output_Location = "D:/CRISM Data/FRT00003E12/Python_Converted/"
#---------------------------------

#***************************************

#Import Statements
#---------------------------------
import rasterio
import numpy as np
import sys
import os
import time
import h5py

#Files Located in other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Support_Files')))
from CRISM_Products_Corrected import * #Located in the Support_Files directory
from support_functions import * #Located in the Support_Files directory
h5_file_path = os.path.join(Output_Location, "stacked_frames.h5")

# Function to apply summary products generation
def generate_summary_products(pixel_class, Summary_Products):

    """
    The formulas for the summary products is contained in the file CRISM_Products_Corrected.py along with the list
    of summary products that there are formulas for. For each pixel, this function calculates those summary products
    by dynamically calling the function name.

    Inputs: pixel_class - multidimensional array containing spectral data from which the summary products are derived.
            Summary_Products - a list of the summary products that need to be calculated.

    Outputs: A list of the output of those summary products for this pixel.

    Note: Requires access to:
        from CRISM_Products_Corrected import *  # Located in the Support_Files directory

    """

    results = []

    for pdct in Summary_Products:
        function_name = f"Generate_{pdct}"
        func = globals().get(function_name)  # Dynamically fetch function by name
        results.append(func(pixel_class))  # Apply the function

    return results

#---------------------------------
# IMPORT TER3 DATA & EXTRACT FREQUENCIES & MASK
#---------------------------------
TER3_Data = rasterio.open(TER3_Location)
tags = TER3_Data.tags()
frequencies = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])
TER3_Data = TER3_Data.read()

#Apply a mask to all pixels with bad data
mask = (TER3_Data == 65535)
count_65535 = np.sum(mask, axis=0)
valid_counts = count_65535[count_65535 != 545]
percentile_95 = np.percentile(valid_counts, 95)
mask = count_65535 <= percentile_95
row_indices, col_indices = np.where(mask)

#---------------------------------
#SETUP H5 FILE
#---------------------------------
#Obtain a list of summary products to calculate
Summary_Products = CRISM_product_list()

if os.path.exists(h5_file_path):
    with h5py.File(h5_file_path, "r") as h5_file:
        stacked_frames = h5_file["stacked_frames"][:]

    # Identify pixels that are still 65535 (only one band needs to be checked)
    valid_mask = stacked_frames[0] == 65535 # Shape: (height, width)

    # Update pixel coordinates: Keep only those that still need processing
    pixel_coords = [(r, c) for r, c in zip(row_indices, col_indices) if valid_mask[r, c]]

else:
    # If no file exists, initialize the stacked_frames array with 65535
    stacked_frames = np.full((len(Summary_Products), TER3_Data.shape[1], TER3_Data.shape[2]), 65535,
                             dtype=np.float32)

    # Precompute the row and column indices
    pixel_coords = list(zip(row_indices, col_indices))

#---------------------------------
#CREATE SUMMARY PRODUCTS
#---------------------------------

# Variables for time tracking
start_time = time.time()  # Record the start time

# Process each pixel in the pixel_coords
for i, (r, c) in enumerate(pixel_coords):

    # Extract spectral data directly from TER3_Data (no need to read it again)
    spectral_data = TER3_Data[:, r, c]

    # Create DataFrame directly without rounding frequencies
    tmp_df = pd.DataFrame({"Wavelength": frequencies, "Numerator": spectral_data})

    # Drop bad data
    tmp_df = tmp_df[tmp_df["Numerator"] != 65535].reset_index(drop=True)

    # Create pixel class for broader function use
    pixel_class = Data_Package(tmp_df)

    # Generate summary products
    summary_results = generate_summary_products(pixel_class, Summary_Products)

    # Store results in the stacked_frames
    stacked_frames[:, r, c] = summary_results

    #Provide user update as to processing progress (easy 100 pixels)
    if i % 100 == 0 and i > 0:
        elapsed_time = time.time() - start_time
        avg_time_per_100 = elapsed_time / (i / 100)
        remaining_iterations = len(pixel_coords) - i
        remaining_time = avg_time_per_100 * (remaining_iterations / 100)

        # Convert remaining time to minutes
        remaining_time_minutes = remaining_time / 60

        print(f"Progress: {i / len(pixel_coords) * 100:.2f}%")
        print(f"Time for 100 iterations: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {remaining_time_minutes:.2f} minutes")

    # After every 1000 calculations, save stacked_frames
    if i % 1000 == 0 and i > 0:

        # Construct the full file path for saving the data
        file_path = os.path.join(Output_Location, f"stacked_frames.h5")

        # Save stacked_frames to the HDF5 file
        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("stacked_frames", data=stacked_frames)

        print(f"Saved stacked_frames_{i}.h5 to {file_path}")

