import numpy as np

class PolynomialRegression:
    def __init__(self, x, y, degree=4):
        self.model = np.poly1d(np.polyfit(x, y, degree))

    def predict_distance(self, area):
        return self.model(area)
