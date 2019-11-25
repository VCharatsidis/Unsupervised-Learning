from sklearn.cluster import KMeans
import numpy as np
import os
from Representations.vae_representations import display_reconstructions, display_centroid


def get_centroids(generation, filepath):
    X = string_to_numpy(filepath)

    kmeans = KMeans(n_clusters=10).fit(X)
    print("centers gen: "+str(generation))
    centroids = kmeans.cluster_centers_

    for centroid in range(10):
        file = open("gen_"+str(generation)+"_centroids.txt", "a")
        for i in centroids[centroid]:
            file.writelines(str(i) + ' ')

        file.writelines("\n")
        file.close()
        display_centroid(centroids[centroid][0:2], centroid, generation)


def get_differences(z, centroids_path, reps_path):
    differences = []
    data = string_to_numpy(reps_path)
    centroids = string_to_numpy(centroids_path)

    for datapoint in data:
        current_differences = []

        for centroid in centroids:
            diff = datapoint - centroid

            for i in diff:
                current_differences.append(i)

        for i in datapoint:
            current_differences.append(i)

        current_differences = np.array(current_differences)
        mean_diffs = current_differences.mean()
        std_diffs = current_differences.std()

        result = []

        for i in range(z):
            result.append(current_differences[i])

        result.append(mean_diffs)
        result.append(std_diffs)

        differences.append(result)

    return differences


def write_differences(z, writepath, centroid_path, data_path):
    differences = get_differences(z, centroid_path, data_path)

    for diff in differences:
        file = open(writepath, "a")
        for i in diff:
            file.writelines(str(i) + ' ')

        file.writelines("\n")
        file.close()


def string_to_numpy(filepath):
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    grubb = os.path.join(script_directory, filepath)

    f = open(grubb, "r")

    contents = f.readlines()
    data = []

    for line in contents:
        X = line.split(' ')
        input_x = []

        Z = map(float, X[:-1])

        for i in Z:
            input_x.append(i)

        data.append(input_x)

    return np.array(data)


z = 2
data_path = '..\\Representations\\vae_reps.txt'

gen = 1
get_centroids(gen, data_path)
next_path = "differences_gen_"+str(gen)+".txt"
centroids_path = "gen_"+str(gen)+"_centroids.txt"
write_differences(z, next_path, centroids_path, data_path)
data_path = next_path

gen = 2
next_path = "differences_gen_"+str(gen)+".txt"
centroids_path = "gen_"+str(gen)+"_centroids.txt"
get_centroids(gen, data_path)





