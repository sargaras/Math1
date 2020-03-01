import numpy as np
from Model import AModel


class Sundail(AModel):
    mu_s=132712.43994e+6
    def __init__(self,t0:float,t1:float,SamplingIncrement:float,X:np.array) ->None:
        super().__init__(t0,t1,SamplingIncrement)
        self._X0=X


    def getRight(self, X: np.array, t:float) -> np.array:
        Y=np.zeros(6)
        for i in range(0,3):
            Y[i]=X[i+3]
        X_Coordin=X[0:3]
        X_Dif=X_Coordin * (-self.mu_s*(1./pow(np.sqrt(X_Coordin.dot(X_Coordin)),3)))
        for j in range(3,6):
            Y[j]=X_Dif[j-3]
        return Y


