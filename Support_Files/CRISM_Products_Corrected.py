import numpy as np
from support_functions import *

def Generate_OLINDEX3_CRISM(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    #Define the bands
    anchor_bands = ["R1750", "R2400"]
    ordinary_bands = ["R1080", "R1152", "R1210", "R1250", "R1263", "R1276", "R1330", "R1368", "R1395", "R1427", "R1470"]
    bands = anchor_bands + ordinary_bands

    #Specify the weights
    RB_Weights = {"RB_R1080":0.03, "RB_R1152":0.03, "RB_R1210":0.03, "RB_R1250":0.03, "RB_R1263":0.07, "RB_R1276":0.07,
                  "RB_R1330":0.12, "RB_R1368":0.12, "RB_R1395":0.14, "RB_R1427":0.18, "RB_R1470":0.18}

    # Create dictionary of variables
    band_dict = {}
    wave_dict = {}
    slope_dict = {}
    rb_dict = {}

    # Obtain require variables
    for band in bands:

        # Wave length
        band_wavelength = int(band[1:])

        # Obtain required variables
        band_dict[band + "_7"] = kernal_width(band_wavelength, wavelengths, 7, dataset)

        # Find closest anchor points
        wave_dict["Wave_" + band] = wavelengths[np.argsort(np.abs(wavelengths - band_wavelength))[0]]

    # Obtain required slops & RB Values & OLINDEX3
    #-----------------------
    # Find anchor points
    Wave_1750, Wave_2400 = wave_dict["Wave_R1750"], wave_dict["Wave_R2400"]
    R1750_7, R2400_7 = band_dict["R1750_7"], band_dict["R2400_7"]

    #Create Result
    OLINDEX3 = 0

    for band in ordinary_bands:

        # Find all the slopes
        slope_dict["RC_" + band] = R1750_7[0] + ((R2400_7[0] - R1750_7[0]) / (Wave_2400 - Wave_1750)) * (
                    wave_dict["Wave_" + band] - Wave_1750)

        rb_dict["RB_" + band] = 1 - band_dict[band + "_7"][0] / slope_dict["RC_" + band]

        OLINDEX3 += rb_dict["RB_" + band] * RB_Weights["RB_" + band]

    return OLINDEX3

def Generate_OLINDEX3(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    #Define the bands
    anchor_bands = ["R1750", "R1862"]
    ordinary_bands = ["R1210", "R1250", "R1263", "R1276", "R1330"]
    bands = anchor_bands + ordinary_bands

    #Specify the weights
    RB_Weights = {"RB_R1210":0.1, "RB_R1250":0.1, "RB_R1263":0.2, "RB_R1276":0.2, "RB_R1330":0.4}

    # Create dictionary of variables
    band_dict = {}
    wave_dict = {}
    slope_dict = {}
    rb_dict = {}

    # Obtain require variables
    for band in bands:

        # Wave length
        band_wavelength = int(band[1:])

        # Obtain required variables
        band_dict[band + "_7"] = kernal_width(band_wavelength, wavelengths, 7, dataset)

        # Find closest anchor points
        wave_dict["Wave_" + band] = wavelengths[np.argsort(np.abs(wavelengths - band_wavelength))[0]]

    # Obtain required slops & RB Values & OLINDEX3
    #-----------------------
    # Find anchor points
    Wave_1750, Wave_1862 = wave_dict["Wave_R1750"], wave_dict["Wave_R1862"]
    R1750_7, R1862_7 = band_dict["R1750_7"], band_dict["R1862_7"]

    #Create Result
    OLINDEX3 = 0

    for band in ordinary_bands:

        # Find all the slopes
        slope_dict["RC_" + band] = R1750_7[0] + ((R1862_7[0] - R1750_7[0]) / (Wave_1862 - Wave_1750)) * (
                    wave_dict["Wave_" + band] - Wave_1750)

        rb_dict["RB_" + band] = 1 - band_dict[band + "_7"][0] / slope_dict["RC_" + band]

        OLINDEX3 += rb_dict["RB_" + band] * RB_Weights["RB_" + band]

    return OLINDEX3

def Generate_BD1300(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1080_5 = kernal_width(1080, wavelengths, 5, dataset)
    R1320_15 = kernal_width(1320, wavelengths, 15, dataset)
    R1750_5 = kernal_width(1750, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_1080 = wavelengths[np.argsort(np.abs(wavelengths - 1080))[0]]
    Wave_1320 = wavelengths[np.argsort(np.abs(wavelengths - 1320))[0]]
    Wave_1750 = wavelengths[np.argsort(np.abs(wavelengths - 1750))[0]]

    b = (Wave_1320-Wave_1080)/(Wave_1750-Wave_1080)
    a = 1-b
    rc = a*R1080_5[0] + b*R1750_5[0]

    return 1 - R1320_15[0]/rc

def Generate_LCPINDEX2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1560_7 = kernal_width(1560, wavelengths, 7, dataset)
    R1690_7 = kernal_width(1690, wavelengths, 7, dataset)
    R1750_7 = kernal_width(1750, wavelengths, 7, dataset)
    R1810_7 = kernal_width(1810, wavelengths, 7, dataset)
    R1870_7 = kernal_width(1870, wavelengths, 7, dataset)
    R2450_7 = kernal_width(2450, wavelengths, 7, dataset)

    # Find closest anchor points
    Wave_1560 = wavelengths[np.argsort(np.abs(wavelengths - 1560))[0]]
    Wave_1690 = wavelengths[np.argsort(np.abs(wavelengths - 1690))[0]]
    Wave_1750 = wavelengths[np.argsort(np.abs(wavelengths - 1750))[0]]
    Wave_1810 = wavelengths[np.argsort(np.abs(wavelengths - 1810))[0]]
    Wave_1870 = wavelengths[np.argsort(np.abs(wavelengths - 1870))[0]]
    Wave_2450 = wavelengths[np.argsort(np.abs(wavelengths - 2450))[0]]

    # Slopes RC
    RC_1690 = R1560_7[0] + ((R2450_7[0] - R1560_7[0]) / (Wave_2450 - Wave_1560)) * (Wave_1690 - Wave_1560)
    RC_1750 = R1560_7[0] + ((R2450_7[0] - R1560_7[0]) / (Wave_2450 - Wave_1560)) * (Wave_1750 - Wave_1560)
    RC_1810 = R1560_7[0] + ((R2450_7[0] - R1560_7[0]) / (Wave_2450 - Wave_1560)) * (Wave_1810 - Wave_1560)
    RC_1870 = R1560_7[0] + ((R2450_7[0] - R1560_7[0]) / (Wave_2450 - Wave_1560)) * (Wave_1870 - Wave_1560)

    # RB Variables
    RB1690 = 1 - R1690_7[0] / RC_1690
    RB1750 = 1 - R1750_7[0] / RC_1750
    RB1810 = 1 - R1810_7[0] / RC_1810
    RB1870 = 1 - R1870_7[0] / RC_1870

    return 0.2* RB1690 + 0.2 * RB1750 + 0.3 * RB1810 + 0.3 * RB1870

def Generate_HCPINDEX2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1690_7 = kernal_width(1690, wavelengths, 7, dataset)
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2140_7 = kernal_width(2140, wavelengths, 7, dataset)
    R2230_7 = kernal_width(2230, wavelengths, 7, dataset)
    R2250_7 = kernal_width(2250, wavelengths, 7, dataset)
    R2430_7 = kernal_width(2430, wavelengths, 7, dataset)
    R2460_7 = kernal_width(2460, wavelengths, 7, dataset)
    R2530_7 = kernal_width(2530, wavelengths, 7, dataset)

    # Find closest anchor points
    Wave_1690 = wavelengths[np.argsort(np.abs(wavelengths - 1690))[0]]
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2140 = wavelengths[np.argsort(np.abs(wavelengths - 2140))[0]]
    Wave_2230 = wavelengths[np.argsort(np.abs(wavelengths - 2230))[0]]
    Wave_2250 = wavelengths[np.argsort(np.abs(wavelengths - 2250))[0]]
    Wave_2430 = wavelengths[np.argsort(np.abs(wavelengths - 2430))[0]]
    Wave_2460 = wavelengths[np.argsort(np.abs(wavelengths - 2460))[0]]
    Wave_2530 = wavelengths[np.argsort(np.abs(wavelengths - 2530))[0]]

    # Slopes RC
    RC_2120 = R1690_7[0] + ((R2530_7[0] - R1690_7[0]) / (Wave_2530 - Wave_1690)) * (Wave_2120 - Wave_1690)
    RC_2140 = R1690_7[0] + ((R2530_7[0] - R1690_7[0]) / (Wave_2530 - Wave_1690)) * (Wave_2140 - Wave_1690)
    RC_2230 = R1690_7[0] + ((R2530_7[0] - R1690_7[0]) / (Wave_2530 - Wave_1690)) * (Wave_2230 - Wave_1690)
    RC_2250 = R1690_7[0] + ((R2530_7[0] - R1690_7[0]) / (Wave_2530 - Wave_1690)) * (Wave_2250 - Wave_1690)
    RC_2430 = R1690_7[0] + ((R2530_7[0] - R1690_7[0]) / (Wave_2530 - Wave_1690)) * (Wave_2430 - Wave_1690)
    RC_2460 = R1690_7[0] + ((R2530_7[0] - R1690_7[0]) / (Wave_2530 - Wave_1690)) * (Wave_2460 - Wave_1690)

    # RB Variables
    RB2120 = 1 - R2120_5[0] / RC_2120
    RB2140 = 1 - R2140_7[0] / RC_2140
    RB2230 = 1 - R2230_7[0] / RC_2230
    RB2250 = 1 - R2250_7[0] / RC_2250
    RB2430 = 1 - R2430_7[0] / RC_2430
    RB2460 = 1 - R2460_7[0] / RC_2460

    return 0.1 * RB2120 + 0.1 * RB2140 + 0.15 * RB2230 + 0.3 * RB2250 + 0.2 * RB2430 + 0.15 * RB2460

def Generate_BD1400(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1330_5 = kernal_width(1330, wavelengths, 5, dataset)
    R1395_3 = kernal_width(1395, wavelengths, 3, dataset)
    R1467_5 = kernal_width(1467, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_1330 = wavelengths[np.argsort(np.abs(wavelengths - 1330))[0]]
    Wave_1395 = wavelengths[np.argsort(np.abs(wavelengths - 1395))[0]]
    Wave_1467 = wavelengths[np.argsort(np.abs(wavelengths - 1467))[0]]

    b = (Wave_1395-Wave_1330)/(Wave_1467-Wave_1330)
    a = 1-b
    rc = a*R1330_5[0] + b*R1467_5[0]

    return 1 - R1395_3[0]/rc

def Generate_BD1500_2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1367_5 = kernal_width(1367, wavelengths, 5, dataset)
    R1525_11 = kernal_width(1525, wavelengths, 11, dataset)
    R1808_5 = kernal_width(1808, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_1367 = wavelengths[np.argsort(np.abs(wavelengths - 1367))[0]]
    Wave_1525 = wavelengths[np.argsort(np.abs(wavelengths - 1525))[0]]
    Wave_1808 = wavelengths[np.argsort(np.abs(wavelengths - 1808))[0]]

    b = (Wave_1525-Wave_1367)/(Wave_1808-Wave_1367)
    a = 1-b
    rc = a*R1367_5[0] + b*R1808_5[0]

    return 1 - R1525_11[0]/rc

def Generate_BD1750_2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1690_5 = kernal_width(1690, wavelengths, 5, dataset)
    R1750_3 = kernal_width(1750, wavelengths, 3, dataset)
    R1815_5 = kernal_width(1815, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_1690 = wavelengths[np.argsort(np.abs(wavelengths - 1690))[0]]
    Wave_1750 = wavelengths[np.argsort(np.abs(wavelengths - 1750))[0]]
    Wave_1815 = wavelengths[np.argsort(np.abs(wavelengths - 1815))[0]]

    b = (Wave_1750-Wave_1690)/(Wave_1815-Wave_1690)
    a = 1-b
    rc = a*R1690_5[0] + b*R1815_5[0]

    return 1 - R1750_3[0]/rc

def Generate_BD2100_2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1930_3 = kernal_width(1930, wavelengths, 3, dataset)
    R2132_5 = kernal_width(2132, wavelengths, 5, dataset)
    R2250_3 = kernal_width(2250, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_1930 = wavelengths[np.argsort(np.abs(wavelengths - 1930))[0]]
    Wave_2132 = wavelengths[np.argsort(np.abs(wavelengths - 2132))[0]]
    Wave_2250 = wavelengths[np.argsort(np.abs(wavelengths - 2250))[0]]

    b = (Wave_2132-Wave_1930)/(Wave_2250-Wave_1930)
    a = 1-b
    rc = a*R1930_3[0] + b*R2250_3[0]

    return 1 - R2132_5[0]/rc

def Generate_BD2165(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2165_3 = kernal_width(2165, wavelengths, 3, dataset)
    R2230_3 = kernal_width(2230, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2230 = wavelengths[np.argsort(np.abs(wavelengths - 2230))[0]]

    b = (Wave_2165-Wave_2120)/(Wave_2230 - Wave_2120)
    a = 1-b
    rc = a*R2120_5[0] + b*R2230_3[0]

    return 1 - R2165_3[0]/rc

def Generate_BD2190(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2185_3 = kernal_width(2185, wavelengths, 3, dataset)
    R2250_3 = kernal_width(2250, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2185 = wavelengths[np.argsort(np.abs(wavelengths - 2185))[0]]
    Wave_2250 = wavelengths[np.argsort(np.abs(wavelengths - 2250))[0]]

    b = (Wave_2185-Wave_2120)/(Wave_2250 - Wave_2120)
    a = 1-b
    rc = a*R2120_5[0] + b*R2250_3[0]

    return 1 - R2185_3[0]/rc

def Generate_MIN2200(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2165_3 = kernal_width(2165, wavelengths, 3, dataset)
    R2210_0 = kernal_width(2210, wavelengths, 3, dataset)
    R2350_5 = kernal_width(2350, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2350 = wavelengths[np.argsort(np.abs(wavelengths - 2350))[0]]

    #Calculte BD2165
    b = (Wave_2165- Wave_2120)/(Wave_2350 - Wave_2120)
    a = 1-b
    rc = a*R2120_5[0] + b*R2350_5[0]
    BD2165 = 1 - R2165_3[0] / rc

    # Calculte BD2210
    b = (Wave_2210- Wave_2120)/(Wave_2350 - Wave_2120)
    a = 1-b
    rc = a*R2120_5[0] + b*R2350_5[0]
    BD2210 = 1 - R2210_0[0] / rc

    return np.minimum(BD2165,BD2210)

def Generate_BD2210_2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2165_5 = kernal_width(2165, wavelengths, 5, dataset)
    R2210_5 = kernal_width(2210, wavelengths, 5, dataset)
    R2290_5 = kernal_width(2290, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2290 = wavelengths[np.argsort(np.abs(wavelengths - 2290))[0]]

    b = (Wave_2210 - Wave_2165)/(Wave_2290 - Wave_2165)
    a = 1-b
    rc = a*R2165_5[0] + b*R2290_5[0]

    return 1 - R2210_5[0]/rc

def Generate_D2200(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1815_7 = kernal_width(1815, wavelengths, 7, dataset)
    R2165_5 = kernal_width(2165, wavelengths, 5, dataset)
    R2210_7 = kernal_width(2210, wavelengths, 7, dataset)
    R2230_7 = kernal_width(2230, wavelengths, 7, dataset)
    R2430_7 = kernal_width(2430, wavelengths, 7, dataset)

    # Find closest anchor points
    Wave_1815 = wavelengths[np.argsort(np.abs(wavelengths - 1815))[0]]
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2230 = wavelengths[np.argsort(np.abs(wavelengths - 2230))[0]]
    Wave_2430 = wavelengths[np.argsort(np.abs(wavelengths - 2430))[0]]

    # Slopes RC
    RC_2210 = R1815_7[0] + ((R2430_7[0] - R1815_7[0]) / (Wave_2430 - Wave_1815)) * (Wave_2210 - Wave_1815)
    RC_2230 = R1815_7[0] + ((R2430_7[0] - R1815_7[0]) / (Wave_2430 - Wave_1815)) * (Wave_2230 - Wave_1815)
    RC_2165 = R1815_7[0] + ((R2430_7[0] - R1815_7[0]) / (Wave_2430 - Wave_1815)) * (Wave_2165 - Wave_1815)

    return 1 - ((R2210_7[0]/RC_2210 + R2230_7[0]/RC_2230)/(2*(R2165_5[0]/RC_2165)))

def Generate_BD2230(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2210_3 = kernal_width(2210, wavelengths, 3, dataset)
    R2235_3 = kernal_width(2235, wavelengths, 3, dataset)
    R2252_3 = kernal_width(2252, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2235 = wavelengths[np.argsort(np.abs(wavelengths - 2235))[0]]
    Wave_2252 = wavelengths[np.argsort(np.abs(wavelengths - 2252))[0]]

    b = (Wave_2235 - Wave_2210)/(Wave_2252 - Wave_2210)
    a = 1-b
    rc = a*R2210_3[0] + b*R2252_3[0]

    return 1 - R2235_3[0]/rc

def Generate_BD2250(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2245_7 = kernal_width(2245, wavelengths, 7, dataset)
    R2340_3 = kernal_width(2340, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2245 = wavelengths[np.argsort(np.abs(wavelengths - 2245))[0]]
    Wave_2340 = wavelengths[np.argsort(np.abs(wavelengths - 2340))[0]]

    b = (Wave_2245 - Wave_2120)/(Wave_2340 - Wave_2120)
    a = 1-b
    rc = a*R2120_5[0] + b*R2340_3[0]

    return 1 - R2245_7[0]/rc

def Generate_MIN2250(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2165_5 = kernal_width(2165, wavelengths, 5, dataset)
    R2210_3 = kernal_width(2210, wavelengths, 3, dataset)
    R2265_3 = kernal_width(2265, wavelengths, 3, dataset)
    R2350_5 = kernal_width(2350, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2265 = wavelengths[np.argsort(np.abs(wavelengths - 2265))[0]]
    Wave_2350 = wavelengths[np.argsort(np.abs(wavelengths - 2350))[0]]

    #Calculte BD2165
    b = (Wave_2210- Wave_2165)/(Wave_2350 - Wave_2165)
    a = 1-b
    rc = a*R2165_5[0] + b*R2350_5[0]
    BD2210 = 1 - R2210_3[0] / rc

    # Calculte BD2210
    b = (Wave_2265- Wave_2165)/(Wave_2350 - Wave_2165)
    a = 1-b
    rc = a*R2165_5[0] + b*R2350_5[0]
    BD2265 = 1 - R2265_3[0] / rc

    return np.minimum(BD2210,BD2265)

def Generate_BD2265(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2210_5 = kernal_width(2210, wavelengths, 5, dataset)
    R2265_3 = kernal_width(2265, wavelengths, 3, dataset)
    R2340_5 = kernal_width(2340, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2265 = wavelengths[np.argsort(np.abs(wavelengths - 2265))[0]]
    Wave_2340 = wavelengths[np.argsort(np.abs(wavelengths - 2340))[0]]

    b = (Wave_2265 - Wave_2210)/(Wave_2340 - Wave_2210)
    a = 1-b
    rc = a*R2210_5[0] + b*R2340_5[0]

    return 1 - R2265_3[0]/rc

def Generate_BD2290(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2250_5 = kernal_width(2250, wavelengths, 5, dataset)
    R2290_5 = kernal_width(2290, wavelengths, 5, dataset)
    R2350_5 = kernal_width(2350, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2250 = wavelengths[np.argsort(np.abs(wavelengths - 2250))[0]]
    Wave_2290 = wavelengths[np.argsort(np.abs(wavelengths - 2290))[0]]
    Wave_2350 = wavelengths[np.argsort(np.abs(wavelengths - 2350))[0]]

    b = (Wave_2290 - Wave_2250)/(Wave_2350 - Wave_2250)
    a = 1-b
    rc = a*R2250_5[0] + b*R2350_5[0]

    return 1 - R2290_5[0]/rc

def Generate_D2300(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1815_5 = kernal_width(1815, wavelengths, 5, dataset)
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2170_5 = kernal_width(2170, wavelengths, 5, dataset)
    R2210_5 = kernal_width(2210, wavelengths, 5, dataset)
    R2290_3 = kernal_width(2290, wavelengths, 3, dataset)
    R2320_3 = kernal_width(2320, wavelengths, 3, dataset)
    R2330_3 = kernal_width(2330, wavelengths, 3, dataset)
    R2530_5 = kernal_width(2530, wavelengths, 5, dataset)

    # Find closest anchor points
    Wave_1815 = wavelengths[np.argsort(np.abs(wavelengths - 1815))[0]]
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2170 = wavelengths[np.argsort(np.abs(wavelengths - 2170))[0]]
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2290 = wavelengths[np.argsort(np.abs(wavelengths - 2290))[0]]
    Wave_2320 = wavelengths[np.argsort(np.abs(wavelengths - 2320))[0]]
    Wave_2330 = wavelengths[np.argsort(np.abs(wavelengths - 2330))[0]]
    Wave_2530 = wavelengths[np.argsort(np.abs(wavelengths - 2530))[0]]

    # Slopes RC
    RC_2120 = R1815_5[0] + ((R2530_5[0] - R1815_5[0]) / (Wave_2530 - Wave_1815)) * (Wave_2120 - Wave_1815)
    RC_2170 = R1815_5[0] + ((R2530_5[0] - R1815_5[0]) / (Wave_2530 - Wave_1815)) * (Wave_2170 - Wave_1815)
    RC_2210 = R1815_5[0] + ((R2530_5[0] - R1815_5[0]) / (Wave_2530 - Wave_1815)) * (Wave_2210 - Wave_1815)
    RC_2290 = R1815_5[0] + ((R2530_5[0] - R1815_5[0]) / (Wave_2530 - Wave_1815)) * (Wave_2290 - Wave_1815)
    RC_2320 = R1815_5[0] + ((R2530_5[0] - R1815_5[0]) / (Wave_2530 - Wave_1815)) * (Wave_2320 - Wave_1815)
    RC_2330 = R1815_5[0] + ((R2530_5[0] - R1815_5[0]) / (Wave_2530 - Wave_1815)) * (Wave_2330 - Wave_1815)

    top_line = R2290_3[0]/RC_2290 + R2320_3[0]/RC_2320 + R2330_3[0]/RC_2330
    bottom_line = R2120_5[0]/RC_2120 + R2170_5[0]/RC_2170 + R2210_5[0]/RC_2210

    return 1 - (top_line/bottom_line)

def Generate_BD2355(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2300_5 = kernal_width(2300, wavelengths, 5, dataset)
    R2355_5 = kernal_width(2355, wavelengths, 5, dataset)
    R2450_5 = kernal_width(2450, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2300 = wavelengths[np.argsort(np.abs(wavelengths - 2300))[0]]
    Wave_2355 = wavelengths[np.argsort(np.abs(wavelengths - 2355))[0]]
    Wave_2450 = wavelengths[np.argsort(np.abs(wavelengths - 2450))[0]]

    b = (Wave_2355 - Wave_2300)/(Wave_2450 - Wave_2300)
    a = 1-b
    rc = a*R2300_5[0] + b*R2450_5[0]

    return 1 - R2355_5[0]/rc

def Generate_SINDEX2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2120_5 = kernal_width(2120, wavelengths, 5, dataset)
    R2290_7 = kernal_width(2290, wavelengths, 7, dataset)
    R2400_3 = kernal_width(2400, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_2120 = wavelengths[np.argsort(np.abs(wavelengths - 2120))[0]]
    Wave_2290 = wavelengths[np.argsort(np.abs(wavelengths - 2290))[0]]
    Wave_2400 = wavelengths[np.argsort(np.abs(wavelengths - 2400))[0]]

    b = (Wave_2290 - Wave_2120)/(Wave_2400 - Wave_2120)
    a = 1-b
    rc = a*R2120_5[0] + b*R2400_3[0]

    return 1 - rc/R2290_7[0]

def Generate_MIN2295_2480(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2165_5 = kernal_width(2165, wavelengths, 5, dataset)
    R2295_5 = kernal_width(2295, wavelengths, 5, dataset)
    R2364_5 = kernal_width(2364, wavelengths, 5, dataset)
    R2480_5 = kernal_width(2480, wavelengths, 5, dataset)
    R2570_5 = kernal_width(2570, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2295 = wavelengths[np.argsort(np.abs(wavelengths - 2295))[0]]
    Wave_2364 = wavelengths[np.argsort(np.abs(wavelengths - 2364))[0]]
    Wave_2480 = wavelengths[np.argsort(np.abs(wavelengths - 2480))[0]]
    Wave_2570 = wavelengths[np.argsort(np.abs(wavelengths - 2570))[0]]

    #Calculte BD2295
    b = (Wave_2295 - Wave_2165)/(Wave_2364 - Wave_2165)
    a = 1-b
    rc = a*R2165_5[0] + b*R2364_5[0]
    BD2295 = 1 - R2295_5[0] / rc

    # Calculte BD2480
    b = (Wave_2480 - Wave_2364)/(Wave_2570 - Wave_2364)
    a = 1-b
    rc = a*R2364_5[0] + b*R2570_5[0]
    BD2480 = 1 - R2480_5[0] / rc

    return np.minimum(BD2295,BD2480)

def Generate_MIN2345_2537(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2250_5 = kernal_width(2250, wavelengths, 5, dataset)
    R2345_5 = kernal_width(2345, wavelengths, 5, dataset)
    R2430_5 = kernal_width(2430, wavelengths, 5, dataset)
    R2537_5 = kernal_width(2537, wavelengths, 5, dataset)
    R2602_5 = kernal_width(2602, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2250 = wavelengths[np.argsort(np.abs(wavelengths - 2250))[0]]
    Wave_2345 = wavelengths[np.argsort(np.abs(wavelengths - 2345))[0]]
    Wave_2430 = wavelengths[np.argsort(np.abs(wavelengths - 2430))[0]]
    Wave_2537 = wavelengths[np.argsort(np.abs(wavelengths - 2537))[0]]
    Wave_2602 = wavelengths[np.argsort(np.abs(wavelengths - 2602))[0]]

    #Calculte BD2345
    b = (Wave_2345 - Wave_2250)/(Wave_2430 - Wave_2250)
    a = 1-b
    rc = a*R2250_5[0] + b*R2430_5[0]
    BD2345 = 1 - R2345_5[0] / rc

    # Calculte BD2537
    b = (Wave_2537 - Wave_2430)/(Wave_2602 - Wave_2430)
    a = 1-b
    rc = a*R2430_5[0] + b*R2602_5[0]
    BD2537 = 1 - R2537_5[0] / rc

    return np.minimum(BD2345,BD2537)

def Generate_BD2500_2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R2364_5 = kernal_width(2364, wavelengths, 5, dataset)
    R2480_5 = kernal_width(2480, wavelengths, 5, dataset)
    R2570_5 = kernal_width(2570, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2364 = wavelengths[np.argsort(np.abs(wavelengths - 2364))[0]]
    Wave_2480 = wavelengths[np.argsort(np.abs(wavelengths - 2480))[0]]
    Wave_2570 = wavelengths[np.argsort(np.abs(wavelengths - 2570))[0]]

    b = (Wave_2480 - Wave_2364)/(Wave_2570 - Wave_2364)
    a = 1-b
    rc = a*R2364_5[0] + b*R2570_5[0]

    return 1 - R2480_5[0]/rc

def Generate_R3920(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R3920_5 = kernal_width(3920, wavelengths, 5, dataset)

    return R3920_5[0]

def Generate_BD1435(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1370_3 = kernal_width(1370, wavelengths, 3, dataset)
    R1432_1 = kernal_width(1432, wavelengths, 1, dataset)
    R1470_3 = kernal_width(1470, wavelengths, 3, dataset)

    # Find Closest wavelengths to these values
    Wave_1370 = wavelengths[np.argsort(np.abs(wavelengths - 1370))[0]]
    Wave_1432 = wavelengths[np.argsort(np.abs(wavelengths - 1432))[0]]
    Wave_1470 = wavelengths[np.argsort(np.abs(wavelengths - 1470))[0]]

    b = (Wave_1432 - Wave_1370)/(Wave_1470 - Wave_1370)
    a = 1-b
    rc = a*R1370_3[0] + b*R1470_3[0]

    return 1 - R1432_1[0]/rc

def Generate_BD1900_2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R1850_5 = kernal_width(1850, wavelengths, 5, dataset)
    R1930_5 = kernal_width(1930, wavelengths, 5, dataset)
    R1985_5 = kernal_width(1985, wavelengths, 5, dataset)
    R2067_5 = kernal_width(2067, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_1850 = wavelengths[np.argsort(np.abs(wavelengths - 1850))[0]]
    Wave_1930 = wavelengths[np.argsort(np.abs(wavelengths - 1930))[0]]
    Wave_1985 = wavelengths[np.argsort(np.abs(wavelengths - 1985))[0]]
    Wave_2067 = wavelengths[np.argsort(np.abs(wavelengths - 2067))[0]]

    #Calculate BD1930 Component
    b = (Wave_1930 - Wave_1850)/(Wave_2067 - Wave_1850)
    a = 1-b
    rc = a*R1850_5[0] + b*R2067_5[0]
    BD1930 = 1 - (R1930_5[0]/rc)

    #Calculate BD1985 Component
    b = (Wave_1985 - Wave_1850)/(Wave_2067 - Wave_1850)
    a = 1-b
    rc = a*R1850_5[0] + b*R2067_5[0]
    BD1985 = 1 - (R1985_5[0]/rc)

    return 0.5 * BD1930 + 0.5 * BD1985

def Generate_BD1900R2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    anchor_bands = ["R1850", "R2060"]
    top_bands = ["R1908", "R1914", "R1921", "R1928", "R1934", "R1941"]
    bottom_bands = ["R1862", "R1869", "R1875", "R2112", "R2120", "R2126"]
    bands = anchor_bands + top_bands + bottom_bands

    #Create dictionary of variables
    band_dict = {}
    wave_dict = {}
    slope_dict = {}

    #Obtain require variables
    for band in bands:

        #Wave length
        band_wavelength = int(band[1:])

        # Obtain required variables
        if band not in ["R1850", "R2060"]:
            band_dict[band + "_1"] = kernal_width(band_wavelength, wavelengths, 1, dataset)
        else:
            band_dict[band + "_1"] = kernal_width(band_wavelength, wavelengths, 5, dataset)

        # Find closest anchor points
        wave_dict["Wave_" + band] = wavelengths[np.argsort(np.abs(wavelengths - band_wavelength))[0]]

    #Obtain required slops
    for band in bands:

        #Find anchor points
        Wave_1850, Wave_2060 = wave_dict["Wave_R1850"], wave_dict["Wave_R2060"]
        R1850_1, R2060_1 = band_dict["R1850_1"], band_dict["R2060_1"]

        #Find all the slopes
        slope_dict["RC_" + band] = R1850_1[0] + ((R2060_1[0] - R1850_1[0]) / (Wave_2060 - Wave_1850)) * (wave_dict["Wave_" + band] - Wave_1850)

    #Calulate numerator
    top_sum = 0
    for band in top_bands:
        top_sum += band_dict[band + "_1"][0] / slope_dict["RC_" + band]

    #Calulate denominator
    bottom_sum = 0
    for band in bottom_bands:
        bottom_sum += band_dict[band + "_1"][0] / slope_dict["RC_" + band]

    return 1 - (top_sum/bottom_sum)

def Generate_CINDEX2(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables
    R3450_9 = kernal_width(3450, wavelengths, 9, dataset)
    R3610_11 = kernal_width(3610, wavelengths, 11, dataset)
    R3875_7 = kernal_width(3875, wavelengths, 7, dataset)

    # Find Closest wavelengths to these values
    Wave_3450 = wavelengths[np.argsort(np.abs(wavelengths - 3450))[0]]
    Wave_3610 = wavelengths[np.argsort(np.abs(wavelengths - 3610))[0]]
    Wave_3875 = wavelengths[np.argsort(np.abs(wavelengths - 3875))[0]]

    # Calculate CINDEX2 Component
    b = (Wave_3610 - Wave_3450)/(Wave_3875 - Wave_3450)
    a = 1-b
    rc = a*R3450_9[0] + b*R3875_7[0]

    #Calculate bandwith normally before flipping
    BD3610 = R3610_11[0]/rc

    return 1 - (1/BD3610)

def Generate_MIN2250(dataset):

    # Find wavelengths
    tags = dataset.tags()
    wavelengths = np.array([float(value.split()[0]) for key, value in tags.items() if key.startswith("Band_")])

    # Obtain required variables

    R2165_5 = kernal_width(2165, wavelengths, 5, dataset)
    R2210_3 = kernal_width(2210, wavelengths, 3, dataset)
    R2265_3 = kernal_width(2265, wavelengths, 3, dataset)
    R2350_5 = kernal_width(2350, wavelengths, 5, dataset)

    # Find Closest wavelengths to these values
    Wave_2165 = wavelengths[np.argsort(np.abs(wavelengths - 2165))[0]]
    Wave_2210 = wavelengths[np.argsort(np.abs(wavelengths - 2210))[0]]
    Wave_2265 = wavelengths[np.argsort(np.abs(wavelengths - 2265))[0]]
    Wave_2350 = wavelengths[np.argsort(np.abs(wavelengths - 2350))[0]]

    #Calculte BD2210
    b = (Wave_2210 - Wave_2165)/(Wave_2350 - Wave_2165)
    a = 1-b
    rc = a*R2165_5[0] + b*R2350_5[0]
    BD2210 = 1 - R2210_3[0] / rc

    # Calculte BD2265
    b = (Wave_2265- Wave_2165)/(Wave_2350 - Wave_2165)
    a = 1-b
    rc = a*R2165_5[0] + b*R2350_5[0]
    BD2265 = 1 - R2265_3[0] / rc

    return np.minimum(BD2265,BD2210)

