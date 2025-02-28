#Raw Data directory
#https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/

#Dataset #1 download all the files with "sul164j" from:
#https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2010/2010_034/frt00016438/

#Dataset #2 download all the files with "sul164j" from:
#https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2010/2010_034/frt0001682b/

import rasterio
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt

#Import the dataset
dataset = rasterio.open('frt00016438_07_su164j_mtr3.img')
dataset_2 = rasterio.open('frt0001682b_07_su164j_mtr3.img')

#Basic Info for the first dataset
print('Image filename: {n}\n'.format(n=dataset.name))
print('Number of bands in image: {n}\n'.format(n=dataset.count))
print('Band names: {dt}'.format(dt=dataset.tags()))

#Plot the surface for the first dataset
R770_Data_Set_1 = dataset.read(1)
R770_Data_Mask_Set_1 = np.where(R770_Data_Set_1 < 65535, R770_Data_Set_1, np.nan)
show(R770_Data_Mask_Set_1, transform=dataset.transform, cmap='gray')

#Plot the surface for the second dataset
R770_Data_Set_1 = dataset.read(1)
R770_Data_Set_2 = dataset_2.read(1)
R770_Data_Mask_Set_2 = np.where(R770_Data_Set_2 < 65535, R770_Data_Set_2, np.nan)

#Plot the surface for both images combined
overall_left = min(dataset.bounds.left, dataset_2.bounds.left)
overall_right = max(dataset.bounds.right, dataset_2.bounds.right)
overall_bottom = min(dataset.bounds.bottom, dataset_2.bounds.bottom)
overall_top = max(dataset.bounds.top, dataset_2.bounds.top)
fig, ax = plt.subplots(figsize=(10, 10))
rasterio.plot.show(R770_Data_Mask_Set_1, transform=dataset.transform, cmap='gray', ax=ax)
rasterio.plot.show(R770_Data_Mask_Set_2, transform=dataset_2.transform, cmap='jet', ax=ax, alpha=0.5)
ax.set_xlim(overall_left, overall_right)
ax.set_ylim(overall_bottom, overall_top)
ax.set_title("Combined Geographic Extent")
plt.show()

#Select another band
BD530_2_Data = dataset.read(3)
BD530_2_Data_Mask = np.where(BD530_2_Data < 65535, BD530_2_Data, np.nan)
show(BD530_2_Data_Mask, transform=dataset.transform, cmap='gray')
rasterio.plot.show_hist(BD530_2_Data_Mask, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

#Stack Multiple Bands
BD860_2_Data = dataset.read(7)
BD860_2_Data_Mask = np.where(BD860_2_Data < 65535, BD860_2_Data, np.nan)
BDI1000VIS_Data = dataset.read(10)
BDI1000VIS_Data_Mask = np.where(BDI1000VIS_Data < 65535, BDI1000VIS_Data, np.nan)
Stack = []
Stack = np.array([BD860_2_Data_Mask, BDI1000VIS_Data_Mask])
rasterio.plot.show_hist(Stack, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

#Plot both bands together
fig, ax = plt.subplots()
ax.scatter(np.ndarray.flatten(BD860_2_Data_Mask), np.ndarray.flatten(BDI1000VIS_Data_Mask))
ax.set_xlabel('BD860_2_Data')
ax.set_ylabel('BDI1000VIS_Data')
fig.show()
