import numpy as np


# Aragues R. Distributed algebraic connectivity estimation for undirected graphs with upper and lower bounds[J]. Automatica, 2014.
# --------------------Aragues original algorithm-------------------------
def iterOnceFords(D, Dk):
    Dk1 = np.zeros_like(D)
    N = D.shape[0]
    # for i in range(N):
    #     for j in range(N):
    #         dij = 0
    #         for k in range(N):
    #             if D[i, k] != 0:
    #                 dij += D[i, k] * Dk[k, j]
    #         Dk1[i, j] = dij
    Dk1 = D @ Dk
    return Dk1


# row sum of Ck
def ci(D):
    N = D.shape[0]
    C = D - 1 / N
    return np.sum(np.abs(C), axis=1)


def ciTolambda2Bounds(c, k, beta):
    # max consensus
    infNorm_c = np.max(c)
    N = c.shape[0]
    upperBounds = (1 - infNorm_c ** (1 / k)) / beta
    lowerBounds = (1 - (infNorm_c / np.sqrt(N)) ** (1 / k)) / beta
    return np.array([lowerBounds, upperBounds])


def Step(D, Dk, k, beta):
    """
    return Dk1, (lowerBounds, upperBounds)
    """
    Dk1 = iterOnceFords(D, Dk)
    cks = ci(Dk1)
    return Dk1, ciTolambda2Bounds(cks, k, beta)


# ----------------------- proposed algorithm----------------------
def iterOnceFordsV2(D, Dk_til, connectivity_est, beta):
    Dk1 = np.zeros_like(D)
    N = D.shape[0]
    # for i in range(N):
    #     for j in range(N):
    #         dij = 0
    #         for k in range(N):
    #             if D[i, k] != 0:
    #                 dij += D[i, k] * Dk_til[k, j]
    #         Dk1[i, j] = dij
    # Dk1 = D @ Dk_til
    # return (Dk1 - 1 / N) / (1 - beta * connectivity_est) + 1 / N
    Dk1 = (D - 1 / N) @ (Dk_til - 1 / N)
    r = Dk1 / (1 - beta * connectivity_est) + 1 / N
    return r


def ciTolambda2(ck_til, beta):
    return (1 - np.max(ck_til)) / beta


def StepV2(D, Dk, connectivity_est, beta):
    """
    return Dk1_til, connectivity
    """
    Dk1_til = iterOnceFordsV2(D, Dk, connectivity_est, beta)
    cks_til = ci(Dk1_til)
    return Dk1_til, ciTolambda2(cks_til, beta)


def IterFielder(L, X, Y, connectivity_est, k1, k2, eps):
    dx = k1 * (-L @ X + connectivity_est * X - k2 * Y)
    X1 = X + dx
    Y1 = Y + dx - eps * L @ Y
    return X1, Y1


# Zhang Y, Li S, Weng J. Distributed Estimation of Algebraic Connectivity[J]. IEEE Transactions on Cybernetics, 2022, 52(5): 3047–3056.
# -----------------------Zhang's method------------------------
def IterFielderAndConnectivity(L, X, Y, Z, k1, k2, k3, eps1, eps2):
    dx = -k1 * Y - k3 * Z * X - k2 * L @ X
    X1 = X + dx
    Y1 = Y + dx - eps1 * L @ Y
    Z1 = Z + 2 * dx * X - eps2 * L @ Z
    connectivitys = -k3 / k2 * Z1
    return X1, Y1, Z1, connectivitys


# ---------------------diff version-------------------------
def ciTolambda2Diff(ck1, ck, beta):
    return (1 - np.max(ck1) / np.max(ck)) / beta


def StepV2Diff(D, Dk, ck, beta):
    """
    return Dk1_til, connectivity
    """
    Dk1 = iterOnceFords(D, Dk)
    ck1 = ci(Dk1)
    return Dk1, ck1, ciTolambda2Diff(ck1, ck, beta)
