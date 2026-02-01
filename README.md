# physics-bachelor-thesis-codes
From-scratch numerical codes developed for my Bachelor's Thesis in Physics.

## About the project
This repository contains the numerical codes developed for my Bachelor's Thesis, which was my first large-scale computational physics project. For this work, I learned **C from scratch** and implemented all simulations at a low level, focusing on understanding both the physics and the numerical methods involved. Linear algebra routines (e.g. eigenvalue problems) are handled using the GNU Scientific Library (GSL).

The project studies **soliton dynamics in the Frenkel-Kontorova model**, a paradigmatic discrete nonlinear system with applications in condensed matter physics, friction, and Josephson junction arrays.

## Physical model

The Frenkel-Kontorova (FK) model describes a one-dimensional chain of particles coupled by harmonic springs and subject to an external periodic potential.
Unlike its continuous counterpart (the Sine-Gordon equation), the discrete nature of the FK model gives rise to genuinely new phenomena, such as the **Peierls-Nabarro potential** and **internal modes of solitons**.

In this project, I study **topological solitons** under periodic boundary conditions, both in static equilibrium and under the action of a constant external force.

## Numerical methods
All simulations were implemented in **C++**, with post processing and visualization performed in **python**.

The main numerical techniques used are:
-**Fourth-order Runge-Kutta (RK4)** for time integration
-**Temporal relaxation** using dissipation to obtain equilibrium configurations
-Computation and diagonalization of the **Hessian matrix** to analyze normal modes
-Collective-coordinate methods to extract effective quantities

## What is implemented
The codes allow the study of:
-Stable and unstable equilibrium soliton profiles
-The **Peierls-Nabarro energy barrier** and its dependence on model parameters
- The **effective potential** felt by the soliton
- The **effective mass** of the soliton as a collective excitation
- Normal modes and the presence of a **localized internal mode**
- Soliton dynamics under a **constant external force**, including depinning, hysteresis, and preferred velocities

## Structure of the repository
The repository is organized by physical and numerical tasks, with each module containing standalone C++ codes and examples.

(The repository will be expanded incrementally as the original thesis codes are cleaned, documented, and uploaded.)

## Future work
Several natural extensions of this project were identified but not implemented within the scope of the thesis, including:
- Interaction between **multiple solitons**
- Inclusion of **thermal noise** and stochastic dynamics
- Effective particle descriptions for interacting solitons

These extensions are intended as future developments of the project.

## Reference
M. Santana Poncio, *Introduction to nonlinear physics: solitons in the Frenkel–Kontorova model*,  
Bachelor’s Thesis, Universidad Complutense de Madrid (2025).
