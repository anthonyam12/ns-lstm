"""
	Holds the network (or other base method) used for the ensmeble as
	well as the variance and mean of the dataset the method was trained over. 

	Author: Anthony Morast
	Date: 4/23/2018
"""

class Method(object):
    def __init__(self, network, mean, variance):
        self.network = network
        self.mean = mean
        self.variance = variance

        self.start = 0
        self.end = 0

    def get_distance(self, mean, variance):
        # euclidian distance
        dist = (self.variance - variance)**2 + (self.mean - mean)**2
        dist = dist**(0.5)
        return dist

    def get_prediction(self, x):
        return self.network.predict(x)

    def train(self, x, y):
        self.network.train(x,y)
