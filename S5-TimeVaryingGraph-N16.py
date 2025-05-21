from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from A import TVGraphLoader
from Algorithms import StepV2, IterFielder, IterFielderAndConnectivity
import matplotlib as mpl

TVgraph = TVGraphLoader(
    "laplacians/Random-16-Nanhai1_PIAS_PCMGAdaptive_N16_eps10_windAtmosphere.Wind.SIMD.WindField4dV3_laplacian.csv"
)
IterTimes = len(TVgraph)
N = TVgraph.N

beta = 1 / (2 * N)
k1 = 1 / N
eps = 1 / N
k2 = N  # 0.4


Dk = np.eye(N)
DkV2 = np.eye(N)
connectivity_est = 1
X = np.linspace(0.1, 0.8, N)
Y = X

# baseline, the coefficients of the method is hard to determine.
k1_ = k3_ = 0.2 * N / (N + 1)
k2_ = eps1 = eps2 = 0.2 / (N + 1)
X_ = np.linspace(0.6, 0.8, N)
Y_ = X_
Z_ = X_ * X_ - 1


# store middle values
connectivitys = np.zeros(IterTimes)
connectivitysBaseline = np.zeros((IterTimes, N))
connectivitys[0] = connectivity_est
NormXs = np.zeros((IterTimes, N))
NormXs[0, :] = X / np.linalg.norm(X, ord=2)
eigs = np.zeros(IterTimes)

for i in range(1, IterTimes):
    L = TVgraph.Graph(i)
    D = np.eye(N) - beta * L
    eig, _ = np.linalg.eigh(L)
    eigs[i] = eig[1]

    DkV2, connectivity_est = StepV2(D, DkV2, connectivity_est, beta)
    dmax = np.max(np.diag(L))
    k1 = eps = 1 / (2 * dmax + 0.2 * np.abs(connectivity_est))
    k2 = 1.2 * np.abs(connectivity_est)
    X, Y = IterFielder(L, X, Y, connectivity_est, k1, k2, eps)
    connectivitys[i] = connectivity_est
    NormXs[i, :] = X / np.linalg.norm(X, ord=2)
    # baseline
    X_, Y_, Z_, connectivity_baseline = IterFielderAndConnectivity(
        L, X_, Y_, Z_, k1_, k2_, k3_, eps1, eps2
    )
    connectivitysBaseline[i, :] = connectivity_baseline


f, ax = plt.subplots()
ax: Axes = ax
cmap = mpl.colormaps["YlGnBu"]
colors = cmap(np.linspace(0.3, 1, N))
t = np.arange(IterTimes)
denseDotted = (0, (1, 1))
width = 2

ax.plot(
    t[1:],
    eigs[1:],
    color="#F13484",
    label="$\\lambda_2$",
    linestyle="-.",
    linewidth=width,
)
ax.plot(t, connectivitys, label="proposed", color="#FD5700", linewidth=width)
l = "baseline"
for i in range(N):
    ax.plot(
        t[1:],
        connectivitysBaseline[1:, i],
        color=colors[i],
        label=l,
        linestyle=denseDotted,
        linewidth=width,
        alpha=0.2,
    )
    l = None

ax.set_ylim([0, 6])
ax.set_xlim([0, IterTimes])
ax.legend()
ax.set_ylabel("Algebraic connectivity $\\hat\\lambda_2$")
ax.set_xlabel("Iterations")
path = "data/conn/figures"
# f.savefig(f"{path}/02-lambda2.{ext}", bbox_inches="tight")


f, ax = plt.subplots()
for i in range(N):
    ax.plot(t, NormXs[:, i])
    # ax.axhline(-eigv[i, 1], linestyle="-.")
plt.show()
