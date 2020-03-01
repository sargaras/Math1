from abc import ABC, abstractmethod
import numpy as np

class AModel(ABC):

    def __init__(self, t0: float, t1: float, SamplingIncrement: float) ->None:
        self._SamplingIncrement = SamplingIncrement
        self._t0 = t0
        self._t1 = t1
        self._X0 = np.array([])
        self._Result = np.zeros((0,self._X0.size))
        self._N=0

    @abstractmethod
    def getRight(self, X: np.array, t: float) ->np.array:
        """to calculate the increment"""

    def getInitialConditions(self) ->np.array:
        return self._X0

    def getOrder(self) ->int:
        return self._X0.size

    def getSamplingIncrement(self) ->float:
        return self._SamplingIncrement

    def getT0(self) ->float:
        return self._t0

    def getT1(self) ->float:
        return self._t1

    def getResult(self) ->np.array: 
        return self._Result

    def addResult(self, X: np.array, t:float) ->None:
        local_res=np.zeros(len(X)+1,float)
        local_res[0]=t
        for i in range(len(X)):
            local_res[i+1]=X[i]
        if self._N==0:
            self._Result=local_res
            self._N=self._N+1
        else:
            self._Result = np.vstack((self._Result,local_res))

        


    def clearResults(self) ->None:
        self._Result = np.array([])   # может тут стоит поискать другую функцию, а не создавать с нуля массив



class AIntegrator(ABC):
    def __init__(self) ->None:
        self._Eps = 1e-8

    def setPrecision(self, Eps: float) ->None:
        self._Eps = Eps

    def getPrecision(self) ->float:
        return self._Eps

    @abstractmethod
    def Run(self, Model: AModel):
        """Start integration"""


class TDormandPrince(AIntegrator):
    def __init__(self) ->None:
        super().__init__()
        self.__c = np.array([0, 1./5, 3./10, 4./5, 8./9, 1., 1.],float)
        self.__a = np.array([
            [0.,0.,0.,0.,0.,0.],
            [1. / 5,0.,0.,0.,0.,0.],
            [3. / 40, 9. / 40,0.,0.,0.,0.],
            [44. / 45, -56. / 15, 32. / 9,0.,0.,0.],
            [19372. / 6561, -25360. / 2187, 64448. / 6561, -212. / 729,0.,0.],
            [9017. / 3168, -355. / 33, 46732. / 5247, 49. / 176, -5103. / 18656,0.],
            [35. / 384, 0., 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84]], float)
        self.__b1 = np.array([35. / 384, 0., 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0], float)
        self.__b2 = np.array([5179. / 57600, 0., 7571. / 16695, 393. / 640, -92097. / 339200, 187. / 2100, 1. / 40], float)
        v = 1.
        self.__u = v
        while (1 + v)>1:
            self.__u = v
            v /= 2

    def Run(self, Model: AModel):
        t = Model.getT0()
        t_out = t
        t1 = Model.getT1()
        h_new = Model.getSamplingIncrement()
        e = 0
        X = Model.getInitialConditions()
        K = np.zeros((7,X.size))

        while t < t1:
            h = h_new
            K[0] = Model.getRight(X, t)
            K[1] = Model.getRight(X + K[0] * h * self.__a[1][0], t + self.__c[1] * h)
            K[2] = Model.getRight(X + K[0] * h * self.__a[2][0] + K[1] * self.__a[2][1] * h, t + self.__c[2] * h)
            K[3] = Model.getRight(X + K[0] * h * self.__a[3][0] + K[1] * self.__a[3][1] * h + K[2] * self.__a[3][2] * h, t + self.__c[3] * h)
            K[4] = Model.getRight(X + K[0] * h * self.__a[4][0] + K[1] * self.__a[4][1] * h + K[2] * h * self.__a[4][2] + K[3] * h * self.__a[4][3], t + self.__c[4] * h)
            K[5] = Model.getRight(X + K[0] * h * self.__a[5][0] + K[1] * self.__a[5][1] * h + K[2] * h * self.__a[5][2] + K[3] * h * self.__a[5][3] + K[4] * h * self.__a[5][4], t + self.__c[5] * h)
            K[6] = Model.getRight(X + K[0] * h * self.__a[6][0] + K[1] * self.__a[6][1] * h + K[2] * h * self.__a[6][2] + K[3] * h * self.__a[6][3] + K[4] * h * self.__a[6][4] + K[5] * h * self.__a[6][5], t + self.__c[6] * h)

            X1 = X + K[0] * self.__b1[0] * h + K[1] * self.__b1[1] * h + K[2] * self.__b1[2] * h + K[3] * self.__b1[3] * h + K[4] * self.__b1[4] * h + K[5] * self.__b1[5] * h + K[6] * self.__b1[6] * h
            X2 = X + K[0] * self.__b2[0] * h + K[1] * self.__b2[1] * h + K[2] * self.__b2[2] * h + K[3] * self.__b2[3] * h + K[4] * self.__b2[4] * h + K[5] * self.__b2[5] * h + K[6] * self.__b2[6] * h

            e1 = 0.
            max0 = np.zeros(X.size)
            for i in range(X.size):
                max1 = np.array([0.00001 , abs(X1[i]), abs(X[i]) , 0.5*self.__u/self._Eps])
                max0[i] = max1.max()
                e1+=(h*(X1[i]-X2[i])/max0[i])**2

            e = (e1/X.size)**0.5

            if 5. > ((e / self._Eps) ** 0.2)/0.9:
                min0 = ((e / self._Eps) ** 0.2)/0.9
            else:
                min0 = 5.

            if 0.1 >= min0:
                max2=0.1
            else:
                max2 = min0

            h_new = h / max2

            if e > self._Eps:
                continue

            while (t_out < t + h) and (t_out <= t1):
                theta = (t_out - t) / h
                b = np.zeros(6)

                b[0] = theta * (1 + theta * (-1337. / 480 + theta * (1039. / 360 + theta * (-1163. / 1152))))
                b[1] = 0
                b[2] = 100. * theta * theta * (1054. / 9275 + theta * (-4682. / 27825 + theta * (379. / 5565))) / 3.
                b[3] = -5. * theta * theta * (27. / 40 + theta * (-9. / 5 + theta * (83. / 96))) / 2.
                b[4] = 18225. * theta * theta * (-3. / 250 + theta * (22. / 375 + theta * (-37. / 600))) / 848.
                b[5] = -22. * theta * theta * (-3. / 10 + theta * (29. / 30 + theta * (-17. / 24))) / 7.

                Xout = X
                for j in range(X.size):
                    Xout = Xout+K[j] * (b[j] * h)

                Model.addResult( Xout, t_out )

                t_out += Model.getSamplingIncrement()

            X = X2
            t += h

