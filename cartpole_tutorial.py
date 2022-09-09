import crocoddyl
import matplotlib.pyplot as plt
import numpy as np


class DifferentialActionModelCartpole(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), 1, 6
        )  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.0
        self.m2 = 0.1
        self.l = 0.5
        self.g = 9.81
        self.costWeights = [
            100.0,
            100.0,
            1.0,
            0.1,
            0.01,
            0.001,
        ]  # sin, 1-cos, x, xdot, thdot, f

    def calc(self, data, x, u=None):
        if u is None:
            u = model.unone
        # Getting the state and control variables
        y, th, ydot, thdot = x[0].item(), x[1].item(), x[2].item(), x[3].item()
        f = u[0].item()

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2
        xddot = (f + m2 * c * s * g - m2 * l * s * thdot**2) / mu
        thddot = (c * f / l + m * g * s / l - m2 * c * s * thdot**2) / mu
        data.xout = np.matrix([xddot, thddot]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([s, 1 - c, y, ydot, thdot, f])).T
        data.cost = 0.5 * sum(np.asarray(data.r) ** 2).item()

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


def main():
    cartpoleDAM = DifferentialActionModelCartpole()

    # Creating the cartpole DAM
    cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)

    timeStep = 5e-2
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)

    # Fill the number of knots (T) and the time step (dt)
    x0 = np.matrix([0.0, 3.14, 0.0, 0.0]).T
    T = 10

    terminalCartpole = DifferentialActionModelCartpole()
    terminalCartpoleDAM = crocoddyl.DifferentialActionModelNumDiff(
        terminalCartpole, True
    )
    terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM)

    terminalCartpole.costWeights[0] = 100
    terminalCartpole.costWeights[1] = 100
    terminalCartpole.costWeights[2] = 1.0
    terminalCartpole.costWeights[3] = 100.0
    terminalCartpole.costWeights[4] = 100.0
    terminalCartpole.costWeights[5] = 0.01
    problem = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)

    # Creating the DDP solver
    ddp = crocoddyl.SolverDDP(problem)

    states = []
    controls = []

    N = 20

    # Solving this problem
    for i in range(N):
        # Solve DDP problem
        ddp.solve()

        # Get next state
        xs = problem.rollout(ddp.us)

        # Set next state to current state
        ddp.problem.x0 = xs[1]

        # Store data
        states.append(xs[0])
        controls.append(ddp.us.tolist()[0])

    states_arr = np.stack(states)
    controls_arr = np.stack(controls)
    data = np.hstack((states_arr, controls_arr))
    time = np.arange(N) * timeStep

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(5):
        axs[i].plot(time, data[:, i])

    plt.show()


if __name__ == "__main__":
    main()
