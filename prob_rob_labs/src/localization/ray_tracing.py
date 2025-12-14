#!/usr/bin/env python3
import numpy as np

def preprocess_lidar_observation(scan_msg):
    """
    Extract and preprocess 4 LiDAR rays from sensor data.
    We only use 4 rays (front, left, back, right) to simplify the measurement model
    while still providing enough information for localization in our hallway environment.
    """
    observed_ranges = np.array(scan_msg.ranges)
    sensor_max_range = scan_msg.range_max

    # We only care about the LiDAR rays at the following angles
    ray_indices = [0, 90, 180, 270]
    observed_4_ranges = np.array([observed_ranges[idx] for idx in ray_indices])

    # If sensor sees 'inf' (nothing), treat it as seeing 'max_range'
    # We should not discard the particle without further information
    observed_4_ranges = np.nan_to_num(observed_4_ranges, posinf=sensor_max_range, neginf=0.0, nan=sensor_max_range)
    observed_4_ranges = np.clip(observed_4_ranges, 0.0, sensor_max_range)

    return observed_4_ranges, sensor_max_range


def get_wall_intersection(ray_angle, particle_x, particle_y, x_lim, y_lim):
    """
    Calculate distance to nearest wall intersection for a given ray.
    """
    min_distance = 100.0

    # Check X-axis walls (left and right)
    if np.cos(ray_angle) > 0.01:
        distance_to_wall = (x_lim[1] - particle_x) / np.cos(ray_angle)
        if distance_to_wall > 0: min_distance = min(min_distance, distance_to_wall)
    elif np.cos(ray_angle) < -0.01:
        distance_to_wall = (x_lim[0] - particle_x) / np.cos(ray_angle)
        if distance_to_wall > 0: min_distance = min(min_distance, distance_to_wall)

    # Check Y-axis walls (top and bottom)
    if np.sin(ray_angle) > 0.01:
        distance_to_wall = (y_lim[1] - particle_y) / np.sin(ray_angle)
        if distance_to_wall > 0: min_distance = min(min_distance, distance_to_wall)
    elif np.sin(ray_angle) < -0.01:
        distance_to_wall = (y_lim[0] - particle_y) / np.sin(ray_angle)
        if distance_to_wall > 0: min_distance = min(min_distance, distance_to_wall)

    return min_distance


def get_landmark_intersection(ray_angle, particle_x, particle_y):
    """
    Calculate distance to red box landmark intersection for a given ray.
    """
    min_distance = 100.0

    # The Red Box at (-5, 2)
    landmark_x_min, landmark_x_max = -5.25, -4.75
    landmark_y_min, landmark_y_max = 1.75, 2.25

    # Check X-parallel faces of the box
    if abs(np.cos(ray_angle)) > 0.01:
        for landmark_x in [landmark_x_min, landmark_x_max]:
            distance_candidate = (landmark_x - particle_x) / np.cos(ray_angle)
            if distance_candidate > 0:
                y_intersection = particle_y + distance_candidate * np.sin(ray_angle)
                if landmark_y_min <= y_intersection <= landmark_y_max:
                    min_distance = min(min_distance, distance_candidate)

    # Check Y-parallel faces of the box
    if abs(np.sin(ray_angle)) > 0.01:
        for landmark_y in [landmark_y_min, landmark_y_max]:
            distance_candidate = (landmark_y - particle_y) / np.sin(ray_angle)
            if distance_candidate > 0:
                x_intersection = particle_x + distance_candidate * np.cos(ray_angle)
                if landmark_x_min <= x_intersection <= landmark_x_max:
                    min_distance = min(min_distance, distance_candidate)

    return min_distance


def predict_4_ranges(particle_x, particle_y, particle_theta, x_lim, y_lim):
    """
    Predict LiDAR ranges at 4 directions from particle pose [front, left, back, right]
    Uses ray-tracing to find distances to walls and the red box landmark.
    """
    predicted_ranges = []

    # Angles corresponding to ROS LiDAR indices [0, 90, 180, 270]
    # 0=Front, pi/2=Left, pi=Back, -pi/2=Right
    angles = [0, np.pi/2, np.pi, -np.pi/2]

    for angle_offset in angles:
        ray_angle = particle_theta + angle_offset

        # Find nearest intersection with walls or landmark
        wall_distance = get_wall_intersection(ray_angle, particle_x, particle_y, x_lim, y_lim)
        landmark_distance = get_landmark_intersection(ray_angle, particle_x, particle_y)

        predicted_ranges.append(min(wall_distance, landmark_distance))

    return np.array(predicted_ranges)
