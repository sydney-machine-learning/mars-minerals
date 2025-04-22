These files are used to create and train the Self-Organizing Map (SOM) model.

Training parameters are configured in the som_training.py file.

The output consists of the following CSV files:
- initial_weights.csv – Contains the initial random weights assigned to the SOM model, useful for comparison.
- reshaped_data.csv – Includes each valid pixel from the processed .h5 TER3 data, with 29 summary products as columns.
- reshaped_indices.csv – Records the original pixel locations for each data instance.
- som_locations.csv – Specifies the pixel locations of each SOM neuron, corresponding to entries in som_weights.csv.
- som_weights.csv – Contains the final trained weights for each SOM neuron.
