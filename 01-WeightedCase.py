from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from Algorithms import Step, StepV2, IterFielder
from A import A1
import matplotlib as mpl

ext = "png"
A = A1
L = np.diag(np.sum(A, axis=1)) - A
dmax = np.max(np.diag(L))

N = L.shape[0]
eig, eigv = np.linalg.eigh(L)
# eig.sort()
print(eig)

IterTimes = 600
beta = 1 / (2 * N)
k1 = eps = 1 / N / 4
k2 = N  # 0.4

D = np.eye(N) - beta * L

Dk = np.eye(N)
DkV2 = np.eye(N)
connectivity_est = 0
X = np.linspace(0.1, 0.8, N)
Y = X

connectivitys = np.zeros(IterTimes)
connectivitys[0] = connectivity_est
bounds_list = []
NormXs = np.zeros((IterTimes, N))
Ys = np.zeros((IterTimes, N))
NormXs[0, :] = X / np.linalg.norm(X, ord=2)


for i in range(1, IterTimes):
    Dk, bounds = Step(D, Dk, i, beta)
    DkV2, connectivity_est = StepV2(D, DkV2, connectivity_est, beta)
    k2 = 1.2 * np.abs(connectivity_est)
    k1 = eps = 1 / (2 * dmax + 0.2 * np.abs(connectivity_est))
    X, Y = IterFielder(L, X, Y, connectivity_est, k1, k2, eps)
    connectivitys[i] = connectivity_est
    bounds_list.append(bounds)
    NormXs[i, :] = X / np.linalg.norm(X, ord=2)

print(DkV2 - 1 / N)
print(np.outer(eigv[:, 1], eigv[:, 1]))

path = "data/conn/figures"
f, ax = plt.subplots(figsize=(6, 3))
ax: Axes = ax
bounds = np.array(bounds_list)
t = np.arange(IterTimes)
denseDotted = (0, (1, 1))
width = 2
ax.axhline(
    eig[1], linestyle="-.", color="#F13484", label="$\\lambda_2$", linewidth=width
)
ax.plot(t, connectivitys, color="#335987", label="proposed", linewidth=width)
ax.plot(
    t[1:],
    bounds[:, 0],
    color="#67A5B8",
    label="bounds",
    linestyle=denseDotted,
    linewidth=width,
)
ax.plot(t[1:], bounds[:, 1], color="#67A5B8", linestyle=denseDotted, linewidth=width)
ax.set_ylim([-0.5, 0.8])
ax.set_xlim([0, IterTimes])
# ax.legend()
ax.set_ylabel("Algebraic connectivity $\\hat\\lambda_2$")
ax.set_xlabel("Iterations")
# f.savefig(f"{path}/01-lambda2.{ext}")
# f.tight_layout()
f.patch.set_alpha(0)

f, ax = plt.subplots(figsize=(6, 4))
cmap = mpl.colormaps["YlGnBu"]
colors = cmap(np.linspace(0.3, 1, N))
for i in range(N):
    ax.axhline(-eigv[i, 1], linestyle="-.", color=colors[i])
    ax.plot(t, NormXs[:, i], color=colors[i], label=i)
ax.set_xlim([0, IterTimes])
ax.set_ylim([-0.9, 0.5])
ax.legend(ncol=N)
ax.set_ylabel("Normalized Fiedler vector $\hat v_2/||\hat v_2||$")
ax.set_xlabel("Iterations")
# f.savefig(f"{path}/01-vector.{ext}", bbox_inches="tight", dpi=300)


f, ax = plt.subplots(figsize=(6, 4))
ax.axhline(0, linestyle="-.")
ax.plot(t, connectivitys - eig[1])

print(connectivitys[-1] - eig[1], bounds[-1, 0] - eig[1])
plt.show()
