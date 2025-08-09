import numpy as np
import torch

class SimpleProblem:
    def __init__(self, Q, p, A, G, h, X):
        self.Q = Q
        self.p = p
        self.A = A
        self.G = G
        self.h = h
        self.X = X
        self.Y = None

    def calc_Y(self):
        self.Y = -self.A @ self.X.T
        self.Y = self.Y.T
