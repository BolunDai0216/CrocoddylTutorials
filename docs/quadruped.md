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

### Cost Models

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

