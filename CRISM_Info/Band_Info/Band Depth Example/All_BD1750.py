import pandas as pd
from Data_Processing import*
import matplotlib.pyplot as plt

#This code calculates the BD1750 index for all minerals in the MRO CRISM Type Spectra Library

df = pd.read_csv('Links.csv', header=None)

# Initialize an empty list to store the results
results = []

for idx in df.index:
    url = df.iloc[idx,1]
    Wavelength, Ratio_IF = clean_data(url)

    # Find desired wavelengths
    R_1550 = find_wavelength(1550, Wavelength, Ratio_IF)
    R_1750 = find_wavelength(1750, Wavelength, Ratio_IF)
    R_1815 = find_wavelength(1815, Wavelength, Ratio_IF)
    mid_val = (1815-1750)/(1815-1550)*R_1550 + (1750-1550)/(1815-1550)*R_1815
    reading = 1 - R_1750/mid_val

    print(f"{df.iloc[idx, 0]} Reading: {reading}")

    results.append([df.iloc[idx, 0], reading])

results_df = pd.DataFrame(results, columns=['Value', 'Reading'])
results_df.to_csv('results.csv', index=False)

# Create the graph
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results_df["Value"].tolist(), results_df["Reading"].tolist())

# Hide all x-tick labels and graph the ones for Gypsum and Alunite
ax.set_xticklabels([])
ax.set_xticks([9, 13])
ax.set_xticklabels(['Alunite', 'Gypsum'])

# Add gridlines
ax.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

# Set axis labels
ax.set_xlabel('Other minerals if not specified', fontsize=12, fontweight='bold')
ax.set_ylabel('BD1750 Depth', fontsize=12, fontweight='bold')

# Customize ticks
ax.tick_params(axis='both', which='both', labelsize=10, width=1.5)
ax.xaxis.set_tick_params(length=6)
ax.yaxis.set_tick_params(length=6)

plt.tight_layout()
plt.show()