import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import textwrap
from Data_Processing import*

#This code is just to generate graphs for Gympsum.
#The BD Depth for Gypsum can be calculated at the end by inputting "1 - R_1750/mid_val".

#Import Gypsum Signature
url = "https://pds-geosciences.wustl.edu/mro/mro-m-crism-4-typespec-v1/mrocr_8001/data/crism_typespec_gypsum.tab"
Wavelength, Ratio_IF = clean_data(url)

#Find desired bands
R_1550 = find_wavelength(1550, Wavelength, Ratio_IF)
R_1750 = find_wavelength(1750, Wavelength, Ratio_IF)
R_1815 = find_wavelength(1815, Wavelength, Ratio_IF)
R_Min = np.min([R_1550, R_1750, R_1815])
R_Max = np.max([R_1550, R_1750, R_1815])
R_Range = R_Max *1.05 - R_Min*0.95

#Find unique wavelength measurement (some meansurmenets come from different sensors etc)
bands=[[0]]
new_band_ind = True

for idx in Wavelength.index[1:]:

    if new_band_ind:
        bands[-1].append(idx)
        diff_wave = Wavelength[idx]-Wavelength[idx-1]
        new_band_ind = False
    else:
        new_diff_wave = Wavelength[idx]-Wavelength[idx-1]

        if (new_diff_wave > diff_wave*1.05 or new_diff_wave < diff_wave*0.95):
            bands.append([idx])
        else:
            bands[-1].append(idx)

# Create the graph and plot the spectrums
fig, ax = plt.subplots(figsize=(6, 4))

for tmp_list in bands:
    ax.plot(Wavelength[tmp_list], Ratio_IF[tmp_list], linestyle='-', label='Band', linewidth=2)

# Add a rectangle patch for the region of interest
square = patches.Rectangle((1550, R_Min*0.95), 1815-1550+40, R_Range, edgecolor='blue', facecolor='none', linewidth=1)
ax.add_patch(square)

#Add text above the rectangle
text_x = 1550 + (1815 - 1550) / 2
text_y = R_Min*0.95 + R_Range + 0.02
text = "Signature of interest for Gypsum"
wrapped_text = textwrap.fill(text, width=15)
ax.text(text_x, text_y, wrapped_text, ha='center', va='bottom', fontsize=8, color='black', wrap=True)

# Set the labels and title
ax.set_xlabel('Wavelength (μm)', fontsize=12, fontweight='bold')
ax.set_ylabel('CRISM Ratioed I/F Corrected', fontsize=12, fontweight='bold')
ax.set_title('CRISM Data with Selected Range', fontsize=14, fontweight='bold')

# Add gridlines and adjust their style
ax.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

# Improve tick marks and labels
ax.tick_params(axis='both', which='both', labelsize=10, width=1.5)
ax.xaxis.set_tick_params(length=6)
ax.yaxis.set_tick_params(length=6)

plt.tight_layout()
plt.show()

# Zoom in and create the graph and plot the spectrums
zoom_index = Wavelength[(Wavelength >= (1550 - 20)) & (Wavelength <= (1815 + 20))].index

fig, ax = plt.subplots(figsize=(6, 4))

for tmp_list in bands:
    tmp_list = sorted(list(set(tmp_list) & set(zoom_index)))
    if len(tmp_list) > 0:
        ax.plot(Wavelength[tmp_list], Ratio_IF[tmp_list], linestyle='-', label='Band', linewidth=2)

# Draw a diagonal line between the points of interest
x = [1550, 1815]
y = [R_1550, R_1815]
ax.plot(x, y, 'b-', linewidth=2)
ax.scatter(x, y, s=100, color='red', edgecolors='black')  # Add circles at the endpoints

# Draw an arrow at the point of interest
mid_val = (1815-1750)/(1815-1550)*R_1550 + (1750-1550)/(1815-1550)*R_1815
ax.scatter(1750, mid_val, s=50, color='green', edgecolors='black')  # Add a circle at the midpoint

# Draw the arrow depending on the comparison between R_1750 and mid_val
if R_1750 < mid_val:
    plt.annotate('', xy=(1750, R_1750), xytext=(1750, mid_val),
                 arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.05, width=0.3, headwidth=5))
else:
    plt.annotate('', xy=(1750, mid_val), xytext=(1750, R_1750),
                 arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.05, width=0.3, headwidth=5))

# Set the labels and title
ax.set_xlabel('Wavelength (μm)', fontsize=12, fontweight='bold')
ax.set_ylabel('CRISM Ratioed I/F Corrected', fontsize=12, fontweight='bold')
ax.set_title('Zoomed-in CRISM Data with Selected Range', fontsize=14, fontweight='bold')

# Add gridlines and adjust their style
ax.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

# Improve tick marks and labels
ax.tick_params(axis='both', which='both', labelsize=10, width=1.5)
ax.xaxis.set_tick_params(length=6)
ax.yaxis.set_tick_params(length=6)

plt.tight_layout()
plt.show()