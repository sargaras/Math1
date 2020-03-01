import numpy as np
from Model import AModel


class Sundail(AModel):
    def getRight(self, X: np.array, t:float):
        print(t)
