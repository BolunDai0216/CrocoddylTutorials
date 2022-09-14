# Walkthrough of Quadruped Control with Crocoddyl

## Building Blocks

This section explains how the optimization problem is transcribed into code. The rough hierarchical structure of the building blocks is

```text
─┬─ Quadrupedal Control Problem
 ├─┬─ actionModel at time 1
 │ ├─── costModel
 │ ├─── contactModel
 │ ├─── actuationModel
 │ └─── stateModel
 ├─── actionModel at time 2
 ┊    ...              
 └─── actionModel at time N
```

### Cost Model

The class that represents the cost function is initialized as

```python
costModel = crocoddyl.CostModelSum(multiBodyState, controlDimension)
```

then individual cost terms are added to the `costModel` using

```python
costModel.addCost("costName", individualCostModel, costWeight)
```

this gives us the cost function

$$\mathrm{cost} = \sum_{i = 1}^{n_\mathrm{cost}}{\mathrm{costweight}_i * \mathrm{individualCost}_i}$$

Each `individualCostModel` has two components: a residual function and an activation function. The residual function computes the error between the measured value and the desired value. The activation function then transforms the error value into a scalar value. 

#### Cost Model Example

If we take the widely used least-square (LS) loss as the cost model:

$$\mathrm{LS\ loss} = \sum_{i=1}^{n}{(x_i^d - x_i)^2}$$

with $\mathbf{x}^d = [x_1^d, \cdots, x_n^d]^T$ being the desired value and $\mathbf{x} = [x_1, \cdots, x_n]$ being the measured value. The residual function gives us the the error between $\mathbf{x}^d$ and $\mathbf{x}$, i.e.,

$$\mathrm{residualModel}(\mathbf{x}^d, \mathbf{x}) = \mathbf{x}^d - \mathbf{x}$$

or in code

```python
stateResidual = crocoddyl.ResidualModelState(multiBodyState, targetState, controlDimension)
```

Then the activation function performs the element-wise square and summation on the $\mathrm{residualModel}$, which is the same as computing the square of the L2 norm

$$\mathrm{activationModel}(\mathbf{r}) = \|\mathbf{r}\|_2^2.$$

or in code

```python
stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
```

with `stateWeights` being an array of all one. 

### Contact Model

The contact model defines which part of the robot contacts with the environment through the contact Jacobian $\mathbf{J}_c$ and the contact constraint $\mathbf{J}_c\ddot{\mathbf{q}} + \dot{\mathbf{J}}_c\dot{\mathbf{q}} = 0$. Similar to the cost model case, the contact model is first initialized using

```python
contactModel = crocoddyl.ContactModelMultiple(multiBodyState, controlDimension)
```

then each `individualContactModel` is added to the `contactModel` object, in the case of point contacts the `individualContactModel` is defined as

```python
individualContactModel = crocoddyl.ContactModel3D(multiBodyState, contactFrameID, contactPositionBaumgarteStabilization, controlDimension, baumgarteStabilizationGains)
```

where the `contactPositionBaumgarteStabilization` term seems to be always set to `np.array([0.0, 0.0, 0.0])`. Then, each `individualContactModel` is added using 

```python
contactModel.addContact(contactName, individualContactModel)
```

with `contactName` being a string indentifying the contact.

### Actuation Model

The actuation model defines which joints are actuated. In the case of a quadruped, the base (body of the quadruped) is a floating-base joint (unactuated) in the URDF file. Thus, a `ActuationModelFloatingBase` is used,

```python
actuationModel = crocoddyl.ActuationModelFloatingBase(multiBodyState)
```

### State Model

The state model is defined using the `pinocchio` model

```python
stateModel = crocoddyl.StateMultibody(pinocchioModel)
```

which defines the kinematics and dynamics of the robot.

## Action Model

The `actionModel` is then defined using 

```python
continuousActionModel = crocoddyl.DifferentialActionModelContactFwdDynamics(stateModel, actuationModel, contactModel, costModel, JMinvJt_damping, enable_force)
```

this defines the continous time model which uses the dynamics $\dot{\mathbf{x}} = \mathbf{F}(\mathbf{x}) + \mathbf{G}(\mathbf{x})\mathbf{u}$. It is them numerically integrated

```python
discreteActionModel = crocoddyl.IntegratedActionModelEuler(continuousActionModel, timeStep)
```

which uses the dynamics $\mathbf{x}^\prime = \mathbf{F}^d(\mathbf{x}) + \mathbf{G}^d(\mathbf{x})\mathbf{u}$

## Standing Problem

The standing problem is the optimal control problem whose solution is the torques required to make the quadruped stand with four foot on the ground. The optimal control problem consists of four cost functions: state cost, control cost, friction cone penalty and the state bound penalty. The state cost is constructed using 

```python
stateResidual = crocoddyl.ResidualModelState(multiBodyState, defaultState, controlDimension)
stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
stateReg = crocoddyl.CostModelResidual(multiBodyState, stateActivation, stateResidual)
```

which corresponds to 

$$\displaystyle\frac{1}{2}(\mathbf{x} - \mathbf{x}_{\mathrm{des}})^T\mathbf{Q}(\mathbf{x} - \mathbf{x}_{\mathrm{des}}),$$ 

here we have the relationship $\mathrm{defaultState} = \mathbf{x}_{\mathrm{des}}$, $\mathrm{stateWeights}^2 = \mathbf{Q}$, and $\mathbf{x}$ coming from `multiBodyState`. The control cost is constructed using

```python
ctrlResidual = crocoddyl.ResidualModelControl(multiBodyState, controlDimension)
ctrlReg = crocoddyl.CostModelResidual(multiBodyState, ctrlResidual)
```

this (according to [here](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/classcrocoddyl_1_1CostModelResidualTpl.html#:~:text=%E2%97%86-,CostModelResidualTpl,-()%20%5B2/2)) corresponds to 

$$\displaystyle\frac{1}{2}\mathbf{u}^T\mathbf{u}.$$

The two penalty functions uses the activation model `ActivationModelQuadraticBarrier` which has the following formulation

$$\frac{1}{2}\Big[\min(\mathbf{r} - \mathrm{lowerBound},\ 0)\Big]^2 + \frac{1}{2}\Big[\max(\mathbf{r} - \mathrm{upperBound},\ 0)\Big]^2$$

which is zero within the range of $[\mathrm{lowerBound},\ \mathrm{upperBound}]$ and grows quadratically as it goes outside the bound. For the state bound the formulation is simply

$$\mathrm{lowerStateBound} \leq \mathrm{state} \leq \mathrm{upperStateBound}$$

and the state bound penatly function penalizes states that are outside of the state bound, which implemented with

```python
lowerBound = np.concatenate(
    [self.state.lb[1 : self.state.nv + 1], 
    self.state.lb[-self.state.nv :]]
)
upperBound = np.concatenate(
    [self.state.ub[1 : self.state.nv + 1], 
    self.state.ub[-self.state.nv :]]
)

stateBoundsResidual = crocoddyl.ResidualModelState(multiBodyState, controlDimension)

stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lowerBound, upperBound))

stateBounds = crocoddyl.CostModelResidual(multiBodyState, stateBoundsActivation, stateBoundsResidual)
```

The friction cone constraint (with four facets) has the form of

$$\begin{align*}
\mathbf{f}_\mathrm{min} &\leq \mathbf{f}_{i, z} \leq \mathbf{f}_\mathrm{max}\\
-\mu\mathbf{f}_{i, z} &\leq \mathbf{f}_{i, x} \leq \mu\mathbf{f}_{i, z}\\
-\mu\mathbf{f}_{i, z} &\leq \mathbf{f}_{i, y} \leq \mu\mathbf{f}_{i, z}
\end{align*}$$

which can be written as

$$\begin{bmatrix}
    0\\
    0\\
    \mathbf{f}_\mathrm{min}
\end{bmatrix} \leq \begin{bmatrix}
    1 & 0 & -\mu\\
    0 & 1 & -\mu\\
    0 & 0 & 1\\
\end{bmatrix}\begin{bmatrix}
    \mathbf{f}_{i, x}\\
    \mathbf{f}_{i, y}\\
    \mathbf{f}_{i, z}
\end{bmatrix} \leq \begin{bmatrix}
    0\\
    0\\
    \mathbf{f}_\mathrm{max}
\end{bmatrix}$$

which is in the form of

$$\mathrm{frictionConeLowerBound} \leq \mathrm{A}\boldsymbol{\lambda} \leq \mathrm{frictionConeUpperBound}.$$

The friction cone penalty function is then implemented as

```python
cone = crocoddyl.FrictionCone(
    surfaceOrientationMatrix, 
    surfaceFrictionCoefficient, 
    facetNumber, 
    useInnerApproximation, 
    minNormalForce, 
    maxNormalForce
)

coneResidual = crocoddyl.ResidualModelContactFrictionCone(multiBodyState, contactFrameId, cone, controlDimension)
coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
frictionCone = crocoddyl.CostModelResidual(multiBodyState, coneActivation, coneResidual)
```
