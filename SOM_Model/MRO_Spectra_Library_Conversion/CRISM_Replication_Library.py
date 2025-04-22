#In the file Link.csv are the links to each mineral in the CRISM MRO Spectral Library.
#Using those links, this code imports the spectral data for each mineral and outputs 29 summary products for each
#mineral in the file MRO_Library_Results which is used extensively thereafter.

import pandas as pd
import re
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.getcwd()

# Set Up Paths
# -----------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Support_Files')))
from CRISM_Products_Corrected import * #Located in the parent directory
from support_functions import * #Located in the parent directory

mineral_links = pd.read_csv('Links.csv', header=None)

# Initialize an empty list to store the results
Product_List = CRISM_product_list()
results = pd.DataFrame(columns=['Mineral Name', 'Mineral Group'] + Product_List)

for index, tmp_row in mineral_links.iterrows():

    # Generate a variable for the mineral (clean name)
    cleaned_name = re.sub(r"[ /-]", "_", str(tmp_row[0]))  # Ensure it's a string

    # Add the mineral to the results table
    new_row = pd.DataFrame([[tmp_row[0],tmp_row[1]]+[np.nan] * (results.shape[1]-2)], columns=results.columns, index=[cleaned_name])
    results = pd.concat([results, new_row])

    #Create a new object for that mineral
    globals()[cleaned_name] = Data_Package(clean_data(mineral_links.iloc[index,2]))

    #Create all the indicators
    for ind in Product_List:
        CRISM_Indicator = "Generate_" + ind + "(" + cleaned_name + ")"
        results.loc[cleaned_name,ind] = eval(CRISM_Indicator)

#Export un-scaled results
results.to_csv("unscaled_MRO_Library_Results.csv")

#Scale Parameters
#-----------------------------

#Set anything below 0 to 0
results.iloc[:,2:-1] = results.iloc[:, 2:-1].where(results.iloc[:, 2:-1] >= 0, 0)

#Scale the results
for tmp_col in results.columns[2:]:  # Start from the 3rd column
    col_data = results[tmp_col]

    #Linear Scaling
    col_min = col_data.min()
    col_max = col_data.max()
    scld_col = (col_data-col_min)/(col_max-col_min)
    results[tmp_col] = scld_col

#Create a plot of all these features
plt.figure()
sns.heatmap(results.iloc[:,2:-1].astype(float), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

#Export Table
results.to_csv("MRO_Library_Results.csv")

#For each mineral find the top summary products over 0
reduction_results = results.copy()

for index, tmp_row in reduction_results.iterrows():
    filtered_values = tmp_row[2:-1][tmp_row[2:-1] > 0].sort_values(ascending=False)
    new_row = pd.Series([0] * len(tmp_row), index=tmp_row.index)
    new_row[filtered_values.head(3).index]=1
    reduction_results.loc[index, results.columns[2:-1]] = new_row

plt.figure()
sns.heatmap(reduction_results.iloc[:,2:-1].astype(float), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

grouped_results = reduction_results.groupby('Mineral Group')
num_groups = len(grouped_results)
fig, axes = plt.subplots(1, num_groups, figsize=(5*num_groups, 5))

for ax, (group_name, group_data) in zip(axes, grouped_results):
    # Create a heatmap for each group
    group_data_filtered = group_data.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
    sns.heatmap(group_data_filtered.T, annot=True, cmap='YlGnBu', cbar=True, ax=ax)

    # Set title for each heatmap
    ax.set_title(f"Group {group_name}")
    ax.set_xlabel('Features')
    ax.set_ylabel('Rows')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()