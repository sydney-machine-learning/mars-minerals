import numpy as np
import pandas as pd

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
        array_list = [dataset.read(int(closest_index) + 1)]  # Read only the closest band
        return [array_list[0], wavelengths[closest_index]]  # Return the reflectance at the closest band

    # Case for hyperspectral data (kernel width is provided)

    # Find the indices of the closest wavelengths
    distances = np.abs(wavelengths - length)
    closest_indices = np.argsort(distances)[:kernel]  # Select the closest 'kernel' number of bands

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
