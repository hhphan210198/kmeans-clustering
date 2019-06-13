import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial.distance import cdist


# generate random data points
def generate_random_numbers(num_count=1000, num_cluster=3):
    x, y = make_blobs(n_samples=num_count, n_features=2, centers=num_cluster)
    return x, y


# traditional initialization
def initialize_centroids(data, num_cluster):
    perm = np.random.permutation(data)
    centroids = perm[:num_cluster, :]
    return centroids


# k-means++ initialization
def initialize_centroids_kmpp(data, num_cluster):
    centroids = np.zeros((num_cluster, data.shape[1]))
    first_centroid_idx = np.random.randint(data.shape[0])
    first_centroid = data[first_centroid_idx]
    centroids[0, :] = first_centroid
    distance_from_nearest_centroids = cdist(first_centroid.reshape(1, -1), data, 'euclidean').T
    # print(distance_from_nearest_centroids)
    centroid_count = 1
    while centroid_count < num_cluster:
        # Choose next centroid using weighted probability distribution
        square_distance = np.square(distance_from_nearest_centroids)
        weighted_distribution = square_distance / np.sum(square_distance)       # Normalize step
        next_centroid_idx = np.random.choice(data.shape[0], p=weighted_distribution.flatten(), replace=False)
        next_centroid = data[next_centroid_idx]
        centroids[centroid_count - 1, :] = next_centroid
        # Update the distance between data points and their nearest centroids
        distance_from_nearest_centroids = np.minimum(distance_from_nearest_centroids, cdist(next_centroid.reshape(1, -1), data, 'euclidean').T)
        centroid_count = centroid_count + 1
    return centroids


# assign data points to their closest centroids
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


# compute new centroids based on the new assignment
def compute_centroids(data, centroid_assignment, num_cluster):
    m, n = data.shape
    sparse_mat = np.zeros((m, num_cluster))
    for i in range(m):
        sparse_mat[i, centroid_assignment[i]] = 1
    count = np.sum(sparse_mat, axis=0, keepdims=True).T
    new_centroid_list = np.divide(np.dot(sparse_mat.T, data), count)
    return new_centroid_list


# compute total cost as sum of euclidean distances between data points and their centroids
def compute_cost(data, centroid_assignment, centroid_list):
    m, n = data.shape
    total_cost = 0
    for i in range(m):
        curr_assignment = centroid_assignment[i]
        total_cost += np.linalg.norm(data[i, :] - centroid_list[curr_assignment, :])
    return total_cost


# Run k-means. Default number of data points = 1000, default number of clusters = 3
def run_k_mean(data, num_cluster=3, tol=1e-7, kmpp=False):

    # Randomly initialize centroids
    if kmpp:
        centroids = initialize_centroids_kmpp(data, num_cluster)
    else:
        centroids = initialize_centroids(data, num_cluster)
    labels = assign_to_closest_centroids(data, centroids)

    cost_hist = []
    # Compute initial cost
    current_cost = compute_cost(data, labels, centroids)
    cost_hist.append(current_cost)

    # Show initial assignment
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
    plt.title("Initial assignment with %d clusters" % num_cluster)
    if kmpp:
        plt.savefig("Plot/Initial_kmpp")
    else:
        plt.savefig("Plot/Initial_vanilla")
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
    plt.title("Final assignment with %d clusters" % num_cluster)
    if kmpp:
        plt.savefig("Plot/Final_kmpp")
    else:
        plt.savefig("Plot/Final_vanilla")
    plt.show()

    return cost_hist


if __name__ == "__main__":
    data, _ = generate_random_numbers(num_cluster=5)
    # Plot of cost vs. iterations
    cost_history = run_k_mean(data, num_cluster=5)
    cost_history_pp = run_k_mean(data, num_cluster=5, kmpp=True)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(cost_history)
    ax[0].set_title("Vanilla")
    ax[0].set_xlabel("Number of iterations")
    ax[0].set_ylabel("Total cost")
    ax[1].plot(cost_history)
    ax[1].set_title("K-means++")
    ax[1].set_xlabel("Number of iterations")
    ax[1].set_ylabel("Total cost")
    plt.show()
