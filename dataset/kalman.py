'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-09-28 02:00:26
'''
import numpy

class Kalman_Filter:
    def __init__(self, A, H, Q, R, z, B = None, impulse = None):
        self._A = A
        self._H = H
        self._Q = Q
        self._R = R
        self._z = z

        self.m = len(z)
        self.n = len(z[0])
        self._identity = numpy.ones([self.n, self.n])

        if (B is None):
            self._B = numpy.zeros([self.n, self.n])
        else:
            self._B = B
        if (impulse is None):
            self._impulse = numpy.zeros([self.m, self.n])
        else:
            self._impulse = impulse

    def __del__(self):
        return

    def _kalman(self, xb, Pb, z, impulse):
        # 测量更新
        tmp = numpy.matmul(Pb, self._H.T)
        K = numpy.matmul(tmp, numpy.linalg.inv(numpy.matmul(self._H, tmp) + self._R))
        x = xb + numpy.matmul(K, (z - numpy.matmul(self._H, xb)))
        P = numpy.matmul((self._identity - numpy.matmul(K, self._H)), Pb)
        # 时间更新
        xb = numpy.matmul(self._A, x) + numpy.matmul(self._B, impulse)
        Pb = numpy.matmul(numpy.matmul(self._A, P), self._A.T) + self._Q
        return x, xb, Pb

    def _kalman1d(self, xb, Pb, z, impulse):
        # 测量更新
        tmp = Pb*self._H
        K = tmp/(self._H*tmp + self._R)
        x = xb + K*(z - self._H*xb)
        P = (1 - K*self._H)*Pb
        # 时间更新
        xb = self._A*x + self._B*impulse
        Pb = self._A*P*self._A + self._Q
        return x, xb, Pb

    def get_filtered_data(self, xb, Pb):
        xx = []
        for i in range(0, self.m):
            if (self.n == 1):
                (x, xb, Pb) = self._kalman1d(xb, Pb, self._z[i], self._impulse[i])
            else:
                (x, xb, Pb) = self._kalman(xb, Pb, self._z[i], self._impulse[i])
            xx.append(x)
        return xx