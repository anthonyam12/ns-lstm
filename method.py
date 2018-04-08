class Method(object):
    def __init__(self, network, mean, variance):
        self.network = network
        self.mean = mean
        self.variance = variance

    def get_distance(self, mean, variance):
        # euclidian distance
        return 999

    def get_prediction(self, x):
        return self.network.predict(x)
