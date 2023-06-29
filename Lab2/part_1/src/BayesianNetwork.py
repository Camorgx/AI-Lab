import numpy as np


class BayesianNetwork:
    def __init__(self, n_labels=10, n_pixels=784, n_values=2) -> None:
        """
        n_labels: number of labels, 10 for digit recognition
        n_pixels: number of pixels, 784 for 28x28 image
        n_values: number of values for each pixel, 0 for black, 1 for white
        """
        self.n_labels = n_labels
        self.n_pixels = n_pixels
        self.n_values = n_values
        # prior probability
        self.labels_prior = np.zeros(n_labels)
        self.pixels_prior = np.zeros((n_pixels, n_values))
        # conditional probability
        self.pixels_cond_label = np.zeros((n_pixels, n_values, n_labels))

    # fit the model with training data
    def fit(self, pixels, labels):
        """
        pixels: (n_samples, n_pixels, )
        labels: (n_samples, )
        """
        n_samples = len(labels)
        # calculate prior probability
        labels_cnt = np.array([np.sum(labels == i) for i in range(10)], dtype=np.float64)
        self.labels_prior = labels_cnt / n_samples
        black_sum = np.zeros(self.n_pixels, dtype=np.uint16)
        for sample in pixels:
            black_sum += sample
        white_sum = np.full(self.n_pixels, fill_value=n_samples, dtype=np.uint16) - black_sum
        self.pixels_prior = np.array([white_sum, black_sum], dtype=np.float64).transpose()
        self.pixels_prior /= n_samples
        # calculate conditional probability
        d = np.zeros((self.n_labels, self.n_pixels), dtype=np.float64)
        for i in range(n_samples):
            d[labels[i]] += pixels[i]
        d = d.transpose()  # d[i][j] = P(p_i = 1 | d = j)
        total = np.tile(labels_cnt, self.n_pixels).reshape((self.n_pixels, self.n_labels))
        d1 = total - d  # d1[i][j] = P(p_i = 0 | d = j)
        d /= total
        d1 /= total
        cond = np.array([d1, d], dtype=np.float64)  # [i][j][k] = P(p_j = i | d = k)
        self.pixels_cond_label = np.swapaxes(cond, 0, 1)  # [i][j][k] = P(p_i = j | d = k)

    def predict_single(self, pixels):
        cond = [[self.pixels_cond_label[index, pixel, d] for index, pixel in enumerate(pixels)]
                for d in range(self.n_labels)]
        prior = [self.pixels_prior[index, pixel] for index, pixel in enumerate(pixels)]
        cond = np.array(cond, dtype=np.float64)
        prior = np.array(prior, dtype=np.float64)
        for line in cond:
            line /= prior
        evaluate = [np.prod(line) for line in cond]
        evaluate *= self.labels_prior
        return np.argmax(evaluate)

    # predict the labels for new data
    def predict(self, pixels):
        """
        pixels: (n_samples, n_pixels, )
        return labels: (n_samples, )
        """
        return np.array([self.predict_single(sample) for sample in pixels])

    # calculate the score (accuracy) of the model
    def score(self, pixels, labels):
        """
        pixels: (n_samples, n_pixels, )
        labels: (n_samples, )
        """
        n_samples = len(labels)
        labels_pred = self.predict(pixels)
        return np.sum(labels_pred == labels) / n_samples


if __name__ == '__main__':
    # load data
    train_data = np.loadtxt('../data/train.csv', delimiter=',', dtype=np.uint8)
    test_data = np.loadtxt('../data/test.csv', delimiter=',', dtype=np.uint8)
    pixels_train, labels_train = train_data[:, :-1], train_data[:, -1]
    pixels_test, labels_test = test_data[:, :-1], test_data[:, -1]
    # build bayesian network
    bn = BayesianNetwork()
    bn.fit(pixels_train, labels_train)
    print('Correct rate: %f%%' % (bn.score(pixels_test, labels_test) * 100))
