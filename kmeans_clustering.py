import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def generate_random_numbers(num_count, num_cluster):
    x, y = make_blobs(n_samples=num_count, n_features=2, centers=num_cluster)
    return x, y


def initialize_centroids(data, num_cluster):
    perm = np.random.permutation(data)
    centroids = perm[:num_cluster, :]
    return centroids


def assign_to_closest_centroids(data, centroid_list):
    m, n = data.shape               # number of data points, 2
    k = centroid_list.shape[0]      # number of centroids
    centroid_assignment = np.zeros(m, dtype=np.int_)
    for i in range(m):
        min_distance = np.linalg.norm(data[i, :] - centroid_list[0, :])
        for j in range(1, k):
            curr_distance = np.linalg.norm(data[i, :] - centroid_list[j, :])
            if curr_distance < min_distance:
                min_distance = curr_distance
                centroid_assignment[i] = j
    return centroid_assignment


def compute_centroids(data, centroid_assignment, num_cluster):
    m, n = data.shape
    sparse_mat = np.zeros((m, num_cluster))
    for i in range(m):
        sparse_mat[i, centroid_assignment[i]] = 1
    count = np.sum(sparse_mat, axis=0, keepdims=True).T
    new_centroid_list = np.divide(np.dot(sparse_mat.T, data), count)
    return new_centroid_list


def compute_cost(data, centroid_assignment, centroid_list):
    m, n = data.shape
    total_cost = 0
    for i in range(m):
        curr_assignment = centroid_assignment[i]
        total_cost += np.linalg.norm(data[i, :] - centroid_list[curr_assignment, :])
    return total_cost


# Run k-means. Default number of data points = 1000, default number of clusters = 3
def run_k_mean(num_count=1000, num_cluster=3, tol=1e-7):
    data, _ = generate_random_numbers(num_count, num_cluster)

    # Randomly initialize centroids
    centroids = initialize_centroids(data, num_cluster)
    labels = assign_to_closest_centroids(data, centroids)

    cost_hist = []
    # Compute initial cost
    current_cost = compute_cost(data, labels, centroids)
    cost_hist.append(current_cost)

    # Show initial assignment
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
    plt.title("Initial assignment")
    plt.show()

    while True:
        centroids = compute_centroids(data, labels, num_cluster)
        labels = assign_to_closest_centroids(data, centroids)
        new_cost = compute_cost(data, labels, centroids)
        if abs(new_cost - current_cost) > tol:
            current_cost = new_cost
            cost_hist.append(current_cost)
        else:
            break

    # Show final assignment
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
    plt.title("Final assignment")
    plt.show()

    return cost_hist


if __name__ == "__main__":
    cost_history = run_k_mean(num_cluster=10)
    # Plot of cost vs. iterations
    plt.plot(cost_history)
    plt.grid(linewidth=1.0)
    plt.xlabel("Number of iterations")
    plt.ylabel("Total cost")
    plt.show()
