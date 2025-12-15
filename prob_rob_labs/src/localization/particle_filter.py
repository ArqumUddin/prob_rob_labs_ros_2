#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation
from localization.ray_tracing import predict_4_ranges, preprocess_lidar_observation
from localization.motion_model import apply_velocity_motion_model

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        self.num_particles = 1000
        # Alphas: Motion noise parameters [v_err_v, w_err_v, v_err_w, w_err_w]
        # Controls how much the particles spread out when moving. When I was playing around with it, I found that setting alphas [0.05, 0.05, 0.01, 0.01] caused
        # the algorithm to converge really slowly
        self.alphas = [0.05, 0.05, 0.1, 0.1]
        
        # Sigma: Measurement noise (in meters)
        # Controls how strict the filter is. Lower = Needs perfect match.
        self.sigma = 0.1
        
        # Map boundaries which emulate a very long hallway the robot has to travel down
        self.x_lim = [-12.0, 12.0]
        self.y_lim = [-3.0, 3.0]

        # State Vector: [x, y, theta, weight]
        self.particles = np.zeros((self.num_particles, 4))
        
        # Global Localization: Scatter particles randomly across the entire map
        self.particles[:, 0] = np.random.uniform(self.x_lim[0], self.x_lim[1], self.num_particles)
        self.particles[:, 1] = np.random.uniform(self.y_lim[0], self.y_lim[1], self.num_particles)
        self.particles[:, 2] = np.random.uniform(-1, 1, self.num_particles)
        
        # Initialize weights uniformly (1/N)
        self.particles[:, 3] = 1.0 / self.num_particles

        self.last_odom = None
        self.last_scan = None
        self.ground_truth = None
        self.last_time = self.get_clock().now()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # Ground truth used ONLY for calculating error metrics, not for localization
        self.create_subscription(ModelStates, '/gazebo/model_states', self.groundtruth_callback, 10)
        
        self.pub_pose = self.create_publisher(PoseStamped, '/pf_pose', 10)
        self.pub_parts = self.create_publisher(PoseArray, '/pf_particles', 10)
        self.pub_uncertainty = self.create_publisher(Float32, '/pf_uncertainty', 10)
        
        # Update happens at 10Hz
        self.create_timer(0.1, self.update)
        self.get_logger().info("Particle Filter Initialized")

    def odom_callback(self, msg): 
        self.last_odom = msg
    
    def scan_callback(self, msg): 
        self.last_scan = msg

    def groundtruth_callback(self, msg):
        """
        Extract ground truth for error analysis
        """
        robot_idx = msg.name.index('waffle_pi')
        robot_pose = msg.pose[robot_idx]
        self.ground_truth = np.array([robot_pose.position.x, robot_pose.position.y])


    def predict_4_ranges(self, particle_x, particle_y, particle_theta):
        """
        Predict LiDAR ranges at 4 cardinal directions from particle pose.
        Uses shared ray-tracing utilities for walls and the red box landmark.
        """
        return predict_4_ranges(particle_x, particle_y, particle_theta, self.x_lim, self.y_lim)

    def motion_model(self, dt):
        """
        Applies Noise to each particle to account for real world motion uncertainty.
        Moves every particle according to odom + noise using arc motion model.
        """
        if self.last_odom is None: return
        linear_vel = self.last_odom.twist.twist.linear.x
        angular_vel = self.last_odom.twist.twist.angular.z

        # We should not add noise if robot is stationary. Adding noise when the robot is stationary may cause
        # the motion model to blow up and never converge.
        if abs(linear_vel) < 1e-3 and abs(angular_vel) < 1e-3: return

        # Apply shared velocity motion model with noise
        self.particles[:, 0], self.particles[:, 1], self.particles[:, 2] = apply_velocity_motion_model(
            self.particles[:, 0], self.particles[:, 1], self.particles[:, 2],
            linear_vel, angular_vel, dt,
            add_noise=True, alphas=self.alphas, num_samples=self.num_particles
        )

    def measurement_model(self):
        """
        Calculates the weights of the particles to determine the importance of each particle.
        Compares real sensor data to predicted particle data.
        """
        if self.last_scan is None: return

        # Use shared preprocessing logic (without max_range validation)
        observed_4_ranges, sensor_max_range = preprocess_lidar_observation(self.last_scan)

        for particle_idx in range(self.num_particles):
            particle_x = self.particles[particle_idx, 0]
            particle_y = self.particles[particle_idx, 1]
            particle_theta = self.particles[particle_idx, 2]

            # Predict what this particle should see
            predicted_4_ranges = predict_4_ranges(particle_x, particle_y, particle_theta, self.x_lim, self.y_lim)

            # If particle predicts a wall at some distance more than max range, clamp to max range to ensure that our error is accurate.
            # In other words, if my sensor sees inf in one direction, and the particle sees a higher number than max range, the particle
            # should not be discarded/given a very low score.
            predicted_4_ranges = np.clip(predicted_4_ranges, 0.0, sensor_max_range)

            # Compute Likelihood (Gaussian)
            total_likelihood = 1.0
            for ray_idx in range(4):
                measurement_error = observed_4_ranges[ray_idx] - predicted_4_ranges[ray_idx]
                # P(z|x) = exp(-error^2 / 2sigma^2)
                total_likelihood *= np.exp(-(measurement_error**2) / (2 * self.sigma**2))

            self.particles[particle_idx, 3] *= total_likelihood

        # Normalize weights so they sum to 1. This is meant to give more meaning to all the particles
        # And allow us to understand how important each weight is within the whole particle space, which will re-inforce
        # our decision to prioritize/keep/discard that weight
        total_weight = np.sum(self.particles[:, 3])
        if total_weight > 0:
            self.particles[:, 3] /= total_weight
        else:
            # If all particles die (weight=0), reset to uniform particle distribution since the robot (and particle filter)
            # Is experiencing the kidnapped scenario
            self.particles[:, 3] = 1.0 / self.num_particles

    def resample(self):
        """
        Low Variance Resampling. I have taken this algorithm from Probabilistic Robotics, Page 110, Table 4.4
        """
        resampled_particles = np.zeros_like(self.particles)
        weights = self.particles[:, 3]
        number_of_particles = self.num_particles
        random = np.random.uniform(0, 1.0 / number_of_particles)
        cumulative_weight = weights[0]
        
        # Current particle index
        i = 0
        for m in range(number_of_particles):
            U = random + m * (1.0 / number_of_particles)
            while U > cumulative_weight:
                i += 1
                cumulative_weight += weights[i]
            
            resampled_particles[m] = self.particles[i]
            resampled_particles[m, 3] = 1.0 / number_of_particles

        self.particles = resampled_particles

    def update(self):
        """
        Main Update loop
        """
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # If the time jump is too big, we should just skip the result to keep calculations consistent.
        if dt > 1.0: return

        self.motion_model(dt)
        self.measurement_model()
        
        # effective_n_of_particles measures how "diverse" the weights are.
        # If effective_n_of_particles is low, it means a few particles hold all the weight, which means we should probably resample
        # Because if we don't we are heaving risking future predictions
        effective_n_of_particles = 1.0 / np.sum(self.particles[:, 3]**2)
        if effective_n_of_particles < self.num_particles / 2:
            self.resample()

        # Get averages so we can calculate how off we are to the true location
        mean_x = np.average(self.particles[:, 0], weights=self.particles[:, 3])
        mean_y = np.average(self.particles[:, 1], weights=self.particles[:, 3])
        sin_sum = np.sum(self.particles[:, 3] * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.particles[:, 3] * np.cos(self.particles[:, 2]))
        mean_theta = np.arctan2(sin_sum, cos_sum)

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = current_time.to_msg()
        msg.pose.position.x = mean_x
        msg.pose.position.y = mean_y
        
        quat = Rotation.from_euler('xyz', [0, 0, mean_theta]).as_quat()
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.pub_pose.publish(msg)

        # Publish particle cloud for Rviz and log to console
        particle_spread = np.std(self.particles[:, :2])
        self.pub_uncertainty.publish(Float32(data=float(particle_spread)))

        if self.ground_truth is not None:
            localization_error = np.linalg.norm(np.array([mean_x, mean_y]) - self.ground_truth)
            if self.last_odom is not None and self.ground_truth is not None:
                linear_vel = self.last_odom.twist.twist.linear.x
                angular_vel = self.last_odom.twist.twist.angular.z
                if abs(linear_vel) >= 1e-3 or abs(angular_vel) >= 1e-3:
                    self.get_logger().info(
                        f"[PF] Error: {localization_error:.2f}m | Spread: {particle_spread:.2f}m | effective_n_of_particles: {effective_n_of_particles:.0f}"
                    )

        if self.pub_parts.get_subscription_count() > 0:
            particle_array_msg = PoseArray()
            particle_array_msg.header = msg.header
            for particle_idx in range(0, self.num_particles):
                particle_pose = Pose()
                particle_pose.position.x = self.particles[particle_idx, 0]
                particle_pose.position.y = self.particles[particle_idx, 1]
                particle_array_msg.poses.append(particle_pose)
            self.pub_parts.publish(particle_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()