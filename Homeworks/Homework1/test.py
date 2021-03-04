import unittest
import numpy as np


class MyTestCase(unittest.TestCase):
    def test1(self):
        w = np.ones([4, 1])
        print(w)

    def test2(self):
        a = np.array([np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), np.array([1, 2])]).reshape(4, )
        print(a)

    def test3(self):
        x = np.array([3, 6, 7, 8, 11])
        print(x)
        n = x.shape[0]
        print(n)
        X = np.stack((np.ones(n), x), axis=1)
        print(X)


if __name__ == '__main__':
    unittest.main()
