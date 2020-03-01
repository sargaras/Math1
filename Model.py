from abc import ABC, abstractmethod
import numpy as np

class AModel(ABC):

    def __init__(self, SamplingIncrement: float, t0: float, t1: float, N: float):
        self.__SamplingIncrement = SamplingIncrement
        self.__t0 = t0
        self.__t1 = t1
        self.__N = N
        self.__X0 = np.array(6)

    @abstractmethod
    def getRight(self, X:np.array, t:float):
        """Посчитать преращение"""

    def getOrder(self) ->int:
        return self.__X0.size()

    def getT0(self) ->float:
        return self.__t0

    def getT1(self) ->float:
        return self.__t1

    def clearResults(self) ->None:
        #"00000"
        self.__N = 0


#
# class AIntegrator:
#     __metaclass__ = ABCMeta
