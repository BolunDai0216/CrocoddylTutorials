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