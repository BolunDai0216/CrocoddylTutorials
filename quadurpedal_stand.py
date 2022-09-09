import os
import sys
import time

import crocoddyl
import example_robot_data
import numpy as np
import pinocchio

from quadruped_standing_utils import QuadrupedalStandingProblem

# Loading the anymal model
anymal = example_robot_data.load("anymal")
robot_model = anymal.model
lims = robot_model.effortLimit
lims *= 0.5  # reduced artificially the torque limits
robot_model.effortLimit = lims

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = QuadrupedalStandingProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the initial state of the robot
q0 = robot_model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(robot_model.nv)
x0 = np.concatenate([q0, v0])

# Defining the walking gait parameters
standing = {
    "stepLength": 0.25,
    "stepHeight": 0.25,
    "timeStep": 1e-2,
    "stepKnots": 25,
    "supportKnots": 2,
}

# Setting up the control-limited DDP solver
solver = crocoddyl.SolverBoxDDP(
    gait.createStandingProblem(
        x0,
        standing["stepLength"],
        standing["stepHeight"],
        standing["timeStep"],
        standing["stepKnots"],
        standing["supportKnots"],
    )
)

# Add the callback functions
print("*** SOLVE ***")
cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

display = crocoddyl.GepettoDisplay(
    anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot]
)
solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])

# Solve the DDP problem
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)

display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])

while True:
    display.displayFromSolver(solver)
    time.sleep(2.0)
