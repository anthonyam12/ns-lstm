from method import *

class Ensemble(object):
    def __init__(self):
        self.methods = []

    def get_prediction(self, x, mean, variance):
        prediction = 0
        for method in self.methods:
            method.get_distance(mean, variance)
            prediction += method.get_prediction(x).reshape(1)
            print(method.get_prediction(x).reshape(1), method.variance, method.mean, variance, mean)
            # calc weight based on distance
        # return weighted prediction
        return prediction / len(self.methods)

    def add_method(self, method):
        self.methods.append(method)
