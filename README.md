# heat_equation
Solve the heat equation

# Equation
$$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

# Discretized
$$\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\Delta t} = \alpha \left( \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{\Delta y^2} \right)$$

**Stability Note:** To prevent numerical divergence, ensure that:
$$\Delta t \leq \frac{\Delta x^2 \Delta y^2}{2\alpha(\Delta x^2 + \Delta y^2)}$$

### Energy Conservation
To verify the solver, we perform a global energy balance:
$$E(t) = \int_{\Omega} u(x,y,t) d\Omega$$
The rate of change of energy must equal the net flux $\Phi$ across the boundaries $\partial\Omega$:
$$\frac{dE}{dt} = \oint_{\partial\Omega} \alpha \nabla u \cdot \mathbf{n} ds$$
In our case, since three sides are insulated ($\nabla u \cdot \mathbf{n} = 0$), the energy increases solely based on the gradient at the left boundary.
