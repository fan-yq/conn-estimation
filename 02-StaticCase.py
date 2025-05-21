from matplotlib import pyplot as plt
from A import StringGraph, CircleGraph
from iter_draw import iter_draw

A0 = CircleGraph(20)
A1 = StringGraph(20)


ext = "png"

path = "data/conn/figures"
f1, f2 = iter_draw(A0, 799)
# f1.savefig(f"{path}/02-cycle-lambda2.{ext}", bbox_inches="tight", dpi=300)
# f2.savefig(f"{path}/02-cycle-vector.{ext}", bbox_inches="tight", dpi=300)
# f1, f2 = iter_draw(A1, 499)
# f1.savefig(f"{path}/02-string.{ext}", bbox_inches="tight", dpi=200)
# f2.savefig(f"{path}/02-string-vector.{ext}", bbox_inches="tight", dpi=200)
plt.show()
