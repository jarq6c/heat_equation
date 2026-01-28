# heat_equation
Solve the heat equation

# Equation
$$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

# Discretized
$$\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\Delta t} = \alpha \left( \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{\Delta y^2} \right)$$

**Stability Note:** To prevent numerical divergence, ensure that:
$$\Delta t \leq \frac{\Delta x^2 \Delta y^2}{2\alpha(\Delta x^2 + \Delta y^2)}$$
