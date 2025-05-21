import numpy as np

# l2 = 0.3
A1 = np.array(
    [
        [0, 0.407, 0.501, 0.055, 0.085, 0],
        [0.407, 0, 0.672, 0, 0, 0.207],
        [0.501, 0.672, 0, 0, 0.199, 0.454],
        [0.055, 0, 0, 0, 0.462, 0],
        [0.085, 0, 0.199, 0.462, 0, 0.157],
        [0, 0.207, 0.454, 0, 0.157, 0],
    ]
)
L1 = np.diag(np.sum(A1, axis=1)) - A1

# l2 = 2
A2 = np.array(
    [
        [-0.0, 0.88, 0.829, 0.797, -0.0, -0.0],
        [0.88, -0.0, 0.835, -0.0, -0.0, 0.851],
        [0.829, 0.835, -0.0, 0.871, 0.857, 0.825],
        [0.797, -0.0, 0.871, -0.0, 0.7, -0.0],
        [-0.0, -0.0, 0.857, 0.7, -0.0, 0.748],
        [-0.0, 0.851, 0.825, -0.0, 0.748, -0.0],
    ]
)

L2 = np.diag(np.sum(A2, axis=1)) - A2

N = L1.shape[0]

# L3 = eigv.T @ np.diag([0, 0.1, 0.1, 0.1, 0.1, 0.2]) @ eigv


def StringGraph(N):
    StringGraph1 = np.zeros((N, N))
    for i in range(N - 1):
        StringGraph1[i, i + 1] = 1
        StringGraph1[i + 1, i] = 1
    return StringGraph1


def CircleGraph(N):
    circlegraph = StringGraph(N)
    circlegraph[0, -1] = 1
    circlegraph[-1, 0] = 1
    return circlegraph


def TimeVarying(k, stop=1000):
    A = k / stop * A1 + (1 - k / stop) * A2
    return np.diag(np.sum(A, axis=1)) - A


def TimeVarying2(k, stop=1000):
    normK = k / stop
    a2 = (normK - 1) ** 2
    a1 = 1 - a2
    # A = normK * normK * A1 + (1 - normK * normK) * A2
    A = a1 * A1 + a2 * A2
    return np.diag(np.sum(A, axis=1)) - A


from read_csv_file import read_csv_file_np


class TVGraphLoader:
    def __init__(self, path: str):
        Ls = read_csv_file_np(path, False)
        N = int(np.sqrt(Ls.shape[1]))
        self.Ls = Ls.reshape(Ls.shape[0], N, N)
        self.N = N

    def Graph(self, k):
        return self.Ls[k]

    def __len__(self):
        return self.Ls.shape[0]
