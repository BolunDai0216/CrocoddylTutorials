from pdb import set_trace

import crocoddyl
import example_robot_data
import numpy as np
import pinocchio

from quadruped_walking_utils import QuadrupedalWalkingProblem


def main():
    # Loading the anymal model
    anymal = example_robot_data.load("anymal")

    # Defining the initial state of the robot
    q0 = anymal.model.referenceConfigurations["standing"].copy()
    v0 = pinocchio.utils.zero(anymal.model.nv)
    x0 = np.concatenate([q0, v0 + np.random.random(18) * 5])

    # Setting up the 3d walking problem
    lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
    gait = QuadrupedalWalkingProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

    # Setting up all tasks
    value = {
        "stepLength": 0.25,
        "stepHeight": 0.15,
        "timeStep": 1e-2,
        "stepKnots": 25,
        "supportKnots": 2,
    }

    cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

    # Creating a walking problem
    solver = crocoddyl.SolverFDDP(
        gait.createWalkingProblem(
            x0,
            value["stepLength"],
            value["stepHeight"],
            value["timeStep"],
            value["stepKnots"],
            value["supportKnots"],
        )
    )

    display = crocoddyl.GepettoDisplay(
        anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot]
    )
    solver.setCallbacks(
        [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
    )

    # Solving the problem with the DDP solver
    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.solve(xs, us, 100, False)

    # Display the entire motion
    display = crocoddyl.GepettoDisplay(
        anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot]
    )
    display.displayFromSolver(solver)


if __name__ == "__main__":
    main()
