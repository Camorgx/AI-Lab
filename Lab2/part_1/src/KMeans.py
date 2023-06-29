import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_image(filepath='../data/ustc-cow.png'):
    img = cv2.imread(filepath)  # Replace with the actual path to your image
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class KMeans:
    def __init__(self, k=4, max_iter=10):
        self.k = k
        self.max_iter = max_iter

    # Randomly initialize the centers
    def initialize_centers(self, points):
        """
        points: (n_samples, n_dims,)
        """
        n, d = points.shape

        centers = np.zeros((self.k, d))
        for k in range(self.k):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            centers[k] = points[random_index].mean(axis=0)

        return centers

    # Assign each point to the closest center
    @staticmethod
    def assign_points(centers, points):
        """
        centers: (n_clusters, n_dims,)
        points: (n_samples, n_dims,)
        return labels: (n_samples, )
        """
        labels = [
            np.argmin([np.linalg.norm(point - center, ord=2) for center in centers])
            for point in points
        ]
        return np.array(labels, dtype=np.uint8)

    # Update the centers based on the new assignment of points
    def update_centers(self, labels, points):
        """
        labels: (n_samples, )
        points: (n_samples, n_dims,)
        return centers: (n_clusters, n_dims,)
        """
        n_samples, n_dims = points.shape
        centers = np.zeros((self.k, n_dims), dtype=np.float64)
        cnt = np.zeros(self.k, dtype=np.float64)
        for i in range(n_samples):
            centers[labels[i]] += points[i]
            cnt[labels[i]] += 1
        for i in range(self.k):
            centers[i] /= cnt[i]
        return centers

    # k-means clustering
    def fit(self, points):
        """
        points: (n_samples, n_dims,)
        return centers: (n_clusters, n_dims,)
        """
        centers = self.initialize_centers(points)
        labels = self.assign_points(centers, points)
        new_centers = self.update_centers(labels, points)
        cnt = 0
        while np.linalg.norm(new_centers - centers, ord=2) > 1e-2 and cnt < self.max_iter:
            centers = new_centers
            labels = self.assign_points(centers, points)
            new_centers = self.update_centers(labels, points)
            cnt += 1
        return centers, labels

    def compress(self, img):
        """
        img: (width, height, 3)
        return compressed img: (width, height, 3)
        """
        # flatten the image pixels
        width, height, n_dim = img.shape
        points = img.reshape((-1, img.shape[-1]))
        centers, labels = self.fit(points)
        result = [centers[label] for label in labels]
        return np.array(result, dtype=np.float64).reshape((width, height, n_dim))


if __name__ == '__main__':
    image = read_image(filepath='../data/ustc-cow.png')
    ks = [2, 4, 8, 16, 32]
    for k in ks:
        kmeans = KMeans(k)
        compressed_img = kmeans.compress(image).round().astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(compressed_img)
        plt.title(f'Compressed Image: k = {k}')
        plt.axis('off')
        plt.savefig(f'../output/compressed_image_{k}.png')
        print(f'Finish k = {k}')
