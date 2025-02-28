import pandas as pd
import numpy as np


def clean_data(web_link):

    # Import Mg_Carbonate Signature
    df = pd.read_csv(web_link, delimiter=", ", engine="python")

    # Extract Data
    Wavelength = df.iloc[:, 0] * 1000
    Ratio_IF = df.iloc[:, 1]

    # Remove no data readings (value of 65535)
    bad_data = Ratio_IF[Ratio_IF == 65535].index
    Wavelength = Wavelength.drop(bad_data).reset_index(drop=True)
    Ratio_IF = Ratio_IF.drop(bad_data).reset_index(drop=True)

    # Check for extreme outliers
    median_IF = np.median(Ratio_IF)
    max_IF = np.max(Ratio_IF)

    if (max_IF / median_IF) > 5:
        bad_data = Ratio_IF[Ratio_IF < median_IF * 5].index
        Wavelength = Wavelength.drop(bad_data).reset_index(drop=True)
        Ratio_IF = Ratio_IF.drop(bad_data).reset_index(drop=True)

    return [Wavelength, Ratio_IF]

def find_wavelength(w_lambda, Wavelength, Ratio_IF):

    # Check if the exact wavelength exists
    if Wavelength[Wavelength == w_lambda].count() > 0:

        R = Ratio_IF[w_lambda]

    else:

        #Weighted average
        lambda_id_below = Wavelength[Wavelength < w_lambda].idxmax()
        lambda_id_above = Wavelength[Wavelength > w_lambda].idxmin()

        lambda_below = Wavelength[lambda_id_below]
        R_below = Ratio_IF[lambda_id_below]
        lambda_above = Wavelength[lambda_id_above]
        R_above = Ratio_IF[lambda_id_above]

        R = ((lambda_above-w_lambda)/(lambda_above-lambda_below))*R_below + ((w_lambda-lambda_below)/(lambda_above-lambda_below))*R_above

    return R