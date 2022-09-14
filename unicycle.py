from pdb import set_trace

import crocoddyl
import matplotlib.pyplot as plt
import numpy as np


def plotUnicycle(x, ax):
    sc, delta = 0.1, 0.1
    a, b, th = x[0].item(), x[1].item(), x[2].item()
    c, s = np.cos(th), np.sin(th)
    refs = [
        ax.arrow(
            a - sc / 2 * c - delta * s,
            b - sc / 2 * s + delta * c,
            c * sc,
            s * sc,
            head_width=0.05,
        ),
        ax.arrow(
            a - sc / 2 * c + delta * s,
            b - sc / 2 * s - delta * c,
            c * sc,
            s * sc,
            head_width=0.05,
        ),
    ]
    return refs


def plotUnicycleSolution(xs, figIndex=1, show=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for x in xs:
        plotUnicycle(x, ax)

    # ax.axis([-1.5, 0.5, -1.5, 0.5])
    ax.grid()

    if show:
        plt.show()


def main():
    model = crocoddyl.ActionModelUnicycle()
    data = model.createData()

    model.costWeights = np.matrix([10, 1]).T  # state weight  # control weight

    x0 = np.matrix([-2.0, 1.0, 1.0]).T  # x, y, theta
    T = 20
    problem = crocoddyl.ShootingProblem(x0, [model] * T, model)

    states = []
    controls = []

    # Define DDP problem
    ddp = crocoddyl.SolverDDP(problem)

    for i in range(100):
        # Solve DDP problem
        ddp.solve()

        # Get next state
        xs = problem.rollout(ddp.us)

        # Set next state to current state
        ddp.problem.x0 = xs[1]

        # Store data
        states.append(xs[0])
        controls.append(ddp.us.tolist()[0])

    plotUnicycleSolution(states)


if __name__ == "__main__":
    main()
