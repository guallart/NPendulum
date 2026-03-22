# Multi-Pendulum Rigid Body Simulation

A lightweight physics engine simulation focusing on the chaotic motion of a multi-link rigid pendulum system using Lagrangian mechanics.

## Overview
This project simulates $n$ interconnected rigid rods. Unlike simple particle simulations, this implementation treats the system as a single articulated body, ensuring that constraints (the links) are never broken and energy is conserved according to the laws of classical mechanics.

## Mathematical Foundation

The simulation avoids traditional Newtonian forces ($F=ma$) in favor of the **Lagrangian mechanics** approach, which is far more efficient for constrained systems.

### 1. The Lagrangian ($\mathcal{L}$)
The state of the system is defined by the angles $\theta_1, \theta_2, \dots, \theta_n$. The Lagrangian is the difference between the Total Kinetic Energy ($T$) and the Total Potential Energy ($V$):

$$\mathcal{L} = T - V$$

### 2. Equations of Motion
We solve the **Euler-Lagrange equations** for each joint $i$:

$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{\theta}_i} \right) - \frac{\partial \mathcal{L}}{\partial \theta_i} = 0$$

### 3. Matrix Form
The resulting system of second-order differential equations is organized into a mass matrix form to solve for angular accelerations ($\ddot{\theta}$):

$$M(\theta) \ddot{\theta} + C(\theta, \dot{\theta}) \dot{\theta} + G(\theta) = 0$$

Where:
* **$M(\theta)$**: The Mass Matrix (Inertia).
* **$C(\theta, \dot{\theta})$**: Coriolis and Centrifugal forces.
* **$G(\theta)$**: Gravitational vector.

## Implementation Details

* **Kinematics:** Forward kinematics are used to map angular states to Cartesian coordinates $(x, y)$ for rendering.
* **Integration:** Uses the **Runge-Kutta 4th Order (RK4)** method to update the state. This prevents the "energy drift" common in simpler Euler integration.
* **Constraints:** High-rigidity links with no deformation or damping.
