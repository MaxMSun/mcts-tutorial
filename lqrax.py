# Adapted from lqrax 0.0.6 by Max Muchen Sun (GPLv3).
# https://github.com/maxmsun/lqrax — renamed API; numerical conventions preserved.

from functools import partial

import jax
import jax.numpy as jnp


class LQR:
    def __init__(self, dt, state_dim, control_dim, Q, R):
        self.dt = dt
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.Q = Q
        self.Q_inv = jnp.linalg.inv(Q)
        self.R = R
        self.R_inv = jnp.linalg.inv(R)
        self._state_jacobian = jax.jacfwd(self.dynamics, argnums=0)
        self._control_jacobian = jax.jacfwd(self.dynamics, argnums=1)

    def dynamics(self, state, control):
        raise NotImplementedError("Implement the continuous-time dynamics f(state, control).")

    def dynamics_step(self, state, control):
        k1 = self.dt * self.dynamics(state, control)
        k2 = self.dt * self.dynamics(state + k1 / 2, control)
        k3 = self.dt * self.dynamics(state + k2 / 2, control)
        k4 = self.dt * self.dynamics(state + k3, control)
        next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state, next_state

    @partial(jax.jit, static_argnums=0)
    def simulate_trajectory(self, initial_state, controls):
        _, states = jax.lax.scan(self.dynamics_step, initial_state, controls)
        return states

    def state_jacobian(self, state, control):
        return self._state_jacobian(state, control)

    def control_jacobian(self, state, control):
        return self._control_jacobian(state, control)

    @partial(jax.jit, static_argnums=0)
    def linearize_dynamics(self, initial_state, controls):
        states = self.simulate_trajectory(initial_state, controls)
        A = jax.vmap(self.state_jacobian)(states, controls)
        B = jax.vmap(self.control_jacobian)(states, controls)
        return states, A, B

    def riccati_dynamics_backward(self, P, A, B):
        return P @ A + A.T @ P - P @ B @ self.R_inv @ B.T @ P + self.Q

    def riccati_step_backward(self, P, coefficients):
        A, B = coefficients
        k1 = self.dt * self.riccati_dynamics_backward(P, A, B)
        k2 = self.dt * self.riccati_dynamics_backward(P + k1 / 2, A, B)
        k3 = self.dt * self.riccati_dynamics_backward(P + k2 / 2, A, B)
        k4 = self.dt * self.riccati_dynamics_backward(P + k3, A, B)
        next_P = P + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_P, next_P

    @partial(jax.jit, static_argnums=0)
    def solve_riccati_backward(self, terminal_P, A_reverse, B_reverse):
        """Consume reverse-time coefficients and return P in reverse-time order."""
        _, P_reverse = jax.lax.scan(
            self.riccati_step_backward, terminal_P, (A_reverse, B_reverse)
        )
        return P_reverse

    def affine_dynamics_backward(self, r, P, A, B, reference):
        return (A - B @ self.R_inv @ B.T @ P).T @ r - self.Q @ reference

    def affine_step_backward(self, r, coefficients):
        P, A, B, reference = coefficients
        k1 = self.dt * self.affine_dynamics_backward(r, P, A, B, reference)
        k2 = self.dt * self.affine_dynamics_backward(r + k1 / 2, P, A, B, reference)
        k3 = self.dt * self.affine_dynamics_backward(r + k2 / 2, P, A, B, reference)
        k4 = self.dt * self.affine_dynamics_backward(r + k3, P, A, B, reference)
        next_r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_r, next_r

    @partial(jax.jit, static_argnums=0)
    def solve_affine_backward(self, terminal_r, P_reverse, A_reverse, B_reverse, reference_reverse):
        """Solve the value function's affine term with all inputs in reverse time."""
        _, r_reverse = jax.lax.scan(
            self.affine_step_backward, terminal_r,
            (P_reverse, A_reverse, B_reverse, reference_reverse),
        )
        return r_reverse

    @partial(jax.jit, static_argnums=0)
    def feedback_control(self, state, P, r, B):
        return -self.R_inv @ B.T @ P @ state - self.R_inv @ B.T @ r

    def closed_loop_dynamics(self, state, P, r, A, B):
        return A @ state + B @ self.feedback_control(state, P, r, B)

    def closed_loop_step(self, state, coefficients):
        P, r, A, B = coefficients
        k1 = self.dt * self.closed_loop_dynamics(state, P, r, A, B)
        k2 = self.dt * self.closed_loop_dynamics(state + k1 / 2, P, r, A, B)
        k3 = self.dt * self.closed_loop_dynamics(state + k2 / 2, P, r, A, B)
        k4 = self.dt * self.closed_loop_dynamics(state + k3, P, r, A, B)
        next_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state, next_state

    @partial(jax.jit, static_argnums=0)
    def simulate_closed_loop(self, initial_state, P, r, A, B):
        _, states = jax.lax.scan(self.closed_loop_step, initial_state, (P, r, A, B))
        return states

    @partial(jax.jit, static_argnums=0)
    def solve(self, initial_state, A, B, reference):
        """Take forward-time sequences and return controls, then post-step states."""
        A_reverse, B_reverse = A[::-1], B[::-1]
        terminal_P = jnp.zeros((self.state_dim, self.state_dim))
        P_reverse = self.solve_riccati_backward(terminal_P, A_reverse, B_reverse)
        P = P_reverse[::-1]
        terminal_r = jnp.zeros(self.state_dim)
        r_reverse = self.solve_affine_backward(
            terminal_r, P_reverse, A_reverse, B_reverse, reference[::-1]
        )
        r = r_reverse[::-1]
        states = self.simulate_closed_loop(initial_state, P, r, A, B)
        controls = jax.vmap(self.feedback_control)(states, P, r, B)
        return controls, states
