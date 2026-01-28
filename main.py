import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
nx, ny = 50, 50          # Grid resolution
dx = dy = 0.1            # Distance between cells
alpha = 20.0             # High diffusivity for fast propagation
source_temp = 100.0
initial_temp = 20.0

# Time stepping
# Stability requires dt <= dx^2 / (4 * alpha)
dt = 0.0001              # Small time step for stability
total_sim_time = 1.0     # 1 second of physics
nt = int(total_sim_time / dt)  # 10,000 total math steps

# Animation settings
fps = 30                 # Standard smooth frame rate
total_frames = 30        # 30 frames @ 30fps = 1 second duration
steps_per_frame = nt // total_frames # ~333 math steps per frame

# --- Initialization ---
u = np.full((nx, ny), initial_temp)
u[0, :] = source_temp    # Initial heat source

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(u.T, cmap='hot', origin='lower', 
               extent=[0, nx*dx, 0, ny*dy], vmin=initial_temp, vmax=source_temp)
plt.colorbar(im, label='Temperature')
ax.set_title("Heat Propagation (1s Simulation)")

def update(frame):
    global u
    # Calculate physics for this frame slice
    for _ in range(steps_per_frame):
        un = u.copy()
        rx = alpha * dt / dx**2
        ry = alpha * dt / dy**2

        # Finite Difference Update
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                         rx * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                         ry * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]))

        # Boundary Conditions
        u[-1, :] = u[-2, :]  # Insulator Right
        u[:, -1] = u[:, -2]  # Insulator Top
        u[:, 0] = u[:, 1]    # Insulator Bottom
        u[0, :] = source_temp # Constant Heat Source Left
    
    im.set_array(u.T)
    return [im]

# --- Create and Save ---
ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True)

# Save as 1-second looping GIF
# loop=0 ensures it restarts immediately
ani.save('heat_equation.gif', writer='pillow', fps=fps, savefig_kwargs={'facecolor':'white'})
