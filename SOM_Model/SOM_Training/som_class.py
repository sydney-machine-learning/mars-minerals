import csv
import numpy as np

class SOM(object):

    """
    Self-Organizing Map (SOM)

    Parameters:
        - x (int): Width of the SOM grid.
        - y (int): Height of the SOM grid.
        - input_dim (int): Number of features in input data.
        - learning_rate (float): Initial learning rate.
        - radius (float): Initial neighborhood radius.
        - num_iter (int): Total number of training iterations.
        - int_iter (int): Starting iteration (default: 0).
        - int_weights (np.ndarray): Optional custom initial weights.

    Notes:
        - If you have stopped previous training you can specify at what iteration training was stopped via int_iter so
        training can resume from that point (with consideration to the learning rate and radius etc) and continue
        training the existing weights which are included via int_weights.

    Credits:

    This code was developed based on the article:
    "Implementing Self-Organizing Maps with Python and TensorFlow" by rubikscode.net
    Source: https://rubikscode.net/2021/07/06/implementing-self-organizing-maps-with-python-and-tensorflow/
    Significant modifications and adaptations have been made to suit the specific requirements of this project.
    """

    def __init__(self, x, y, input_dim, learning_rate, radius, num_iter, int_iter=0, int_weights=None):
        self._x = x
        self._y = y
        self._learning_rate = float(learning_rate)
        self._radius = float(radius)
        self._num_iter = num_iter
        self._initial_iteration = int_iter

        self._weights = int_weights if int_weights is not None else np.random.rand(x * y, input_dim)
        self._locations = self._generate_index_matrix(x, y)

        self._initial_weights = np.copy(self._weights)
        self._initial_centroid_matrix = self._create_centroid_matrix(self._initial_weights)

        self._current_learning_rate = float(learning_rate)
        self._current_radius = float(radius)

    def _generate_index_matrix(self, x, y):
        return np.array([[i, j] for i in range(x) for j in range(y)])

    def _create_centroid_matrix(self, weights):
        centroid_matrix = [[] for _ in range(self._x)]
        for i, loc in enumerate(self._locations):
            centroid_matrix[loc[0]].append(weights[i])
        return centroid_matrix

    def train(self, input_vects, cluster_assignments, samples_cluster, burn_in):
        unique_clusters = np.unique(cluster_assignments)
        cluster_ie = samples_cluster // len(unique_clusters)

        if self._initial_iteration < burn_in:
            for burn_no in range(self._initial_iteration, burn_in):
                print(f"Burn In Period: {burn_no}")
                self._sub_training(0, input_vects, cluster_assignments, unique_clusters, cluster_ie)

        for iter_no in range(self._initial_iteration, self._num_iter):
            self._sub_training(iter_no, input_vects, cluster_assignments, unique_clusters, cluster_ie)

            if iter_no % 10 == 0:
                self.export_to_csv()

        self._weights_list = np.copy(self._weights)
        self._centroid_matrix = self._create_centroid_matrix(self._weights_list)

    def _sub_training(self, iter_no, input_vects, cluster_assignments, unique_clusters, cluster_ie):
        self.update_learning_parameters(iter_no)

        selected_rows = []
        for cluster in unique_clusters:
            indices = np.where(cluster_assignments == cluster)[0]
            sampled = np.random.choice(indices, size=min(cluster_ie, len(indices)), replace=False)
            selected_rows.append(input_vects[sampled])

        selected_rows_array = np.vstack(selected_rows)
        np.random.shuffle(selected_rows_array)

        print(f"Iteration {iter_no}, Current Learning Rate: {self._current_learning_rate}, Current Radius: {self._current_radius}")
        print(f"Minimum of column four is {np.min(self._weights[:, 4])}")

        for input_vect in selected_rows_array:
            self._update_weights(input_vect, iter_no)

    def _update_weights(self, input_vect, iter_no):
        bmu_index = self._find_bmu(input_vect)
        bmu_location = self._locations[bmu_index]

        for i in range(self._x * self._y):
            distance = np.linalg.norm(self._locations[i] - bmu_location)
            if distance < self._current_radius:
                neighborhood_factor = np.exp(-distance**2 / (2 * self._current_radius**2))
                self._weights[i] += self._current_learning_rate * neighborhood_factor * (input_vect - self._weights[i])

    def _find_bmu(self, input_vect):
        distances = np.linalg.norm(self._weights - input_vect, axis=1)
        return np.argmin(distances)

    def update_learning_parameters(self, iter_no):
        decay = np.exp((np.log(0.015) / self._num_iter) * iter_no)
        self._current_learning_rate = self._learning_rate * decay
        self._current_radius = self._radius * decay

    def map_input(self, input_vectors):
        return [self._locations[self._find_bmu(vect)] for vect in input_vectors]

    def compute_mid(self):
        mid_values = np.zeros((self._x, self._y))

        for i in range(self._x):
            for j in range(self._y):
                current = np.array(self._centroid_matrix[i][j])
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self._x and 0 <= nj < self._y:
                        neighbors.append(np.array(self._centroid_matrix[ni][nj]))

                if neighbors:
                    distances = [np.linalg.norm(current - neighbor) for neighbor in neighbors]
                    mid_values[i, j] = np.mean(distances)

        return mid_values

    def export_to_csv(self, initial_weights_filename='initial_weights.csv', weights_filename='som_weights.csv', locations_filename='som_locations.csv'):
        def write_csv(filename, header, data):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in data:
                    writer.writerow(row)

        write_csv(initial_weights_filename, [f'Weight {i}' for i in range(self._initial_weights.shape[1])], self._initial_weights)
        print(f"Saved initial weights to {initial_weights_filename}")

        write_csv(weights_filename, [f'Weight {i}' for i in range(self._weights.shape[1])], self._weights)
        print(f"Saved weights to {weights_filename}")

        write_csv(locations_filename, ['X', 'Y'], self._locations)
        print(f"Saved locations to {locations_filename}")