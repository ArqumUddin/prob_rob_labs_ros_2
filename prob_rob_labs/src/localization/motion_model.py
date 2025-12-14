#!/usr/bin/env python3
import numpy as np

def apply_velocity_motion_model(x, y, theta, linear_vel, angular_vel, dt, add_noise=False, alphas=None, num_samples=1):
    """
    Applies velocity motion model with optional noise.
    Uses motion approximation from Probabilistic Robotics Table 5.3.
    Falls back to linear approximation when angular velocity ≈ 0 to avoid division by zero.
    """
    # We should not add noise if alphas is not provided but add_noise is True
    if add_noise and alphas is None:
        raise ValueError("Must provide alphas when add_noise=True")

    # Apply noise if requested (used by Particle Filter for particle diversity)
    if add_noise:
        # Velocity-dependent noise from Probabilistic Robotics, Page 124 Table 5.3
        noisy_linear_vel = linear_vel + np.random.normal(0, np.sqrt(alphas[0] * linear_vel**2 + alphas[1] * angular_vel**2), num_samples)
        noisy_angular_vel = angular_vel + np.random.normal(0, np.sqrt(alphas[2] * linear_vel**2 + alphas[3] * angular_vel**2), num_samples)
    else:
        # No noise (used by UKF which handles uncertainty via covariance propagation)
        noisy_linear_vel = linear_vel
        noisy_angular_vel = angular_vel

    # Use arc motion algorithm IF angular velocity is significant
    # Otherwise use linear approximation to avoid division by zero)
    if abs(angular_vel) < 1e-3:
        # Linear approximation
        new_x = x + noisy_linear_vel * np.cos(theta) * dt
        new_y = y + noisy_linear_vel * np.sin(theta) * dt
        new_theta = theta + noisy_angular_vel * dt
    else:
        # Arc motion
        ratio = noisy_linear_vel / noisy_angular_vel
        new_x = x - ratio * np.sin(theta) + ratio * np.sin(theta + noisy_angular_vel * dt)
        new_y = y + ratio * np.cos(theta) - ratio * np.cos(theta + noisy_angular_vel * dt)
        new_theta = theta + noisy_angular_vel * dt

    return new_x, new_y, new_theta
