from sklearn.cluster import KMeans


def elbow_method(data, num_cluster_ls):
    ssd = []
    for num_cluster in num_cluster_ls:
        km = KMeans(n_clusters=num_cluster).fit(data)
        ssd.append(km.inertia_)
    return ssd
