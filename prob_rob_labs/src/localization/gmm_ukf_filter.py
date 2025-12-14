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

class GaussianMixtureUKF(Node):
    def __init__(self):
        super().__init__('gmm_ukf')
        
        # Number of Gaussian Hypotheses (Particles). 
        # Unlike PF which needs 1000 points, we only need ~20 because each particle here is a full Gaussian distribution (Mean + Covariance)
        # This allows us to track the probability "blobs" rather than raw points, saving computation
        self.num_particles = 20
        
        # Map boundaries which emulate a very long hallway the robot has to travel down
        self.x_lim = [-12.0, 12.0]
        self.y_lim = [-3.0, 3.0]

        # These control how we sample the "Sigma Points" from the Gaussian distribution
        # Spread of sigma points (small = tight cluster around mean)
        self.alpha = 0.1   
        # Prior knowledge of distribution
        self.beta = 2.0    
        # Dimension of state vector (x, y, theta)
        self.L = 3 
        # Scaling factor used for weight calculation.
        self.lamb = (self.alpha**2 * self.L) - self.L
        
        # Q determines how much we trust the Motion Model. 
        # I increased this from 0.1 to 0.2 because the filter was becoming "Overconfident".
        # If Q is too small, the filter thinks it knows exactly where it is and ignores the sensors.
        # By increasing this, I am essentially telling the filter that it should trust the sensors more
        self.Q = np.diag([0.2**2, 0.2**2, 0.1**2])
        
        # R determines how much we trust the Sensors
        # This acts like 'sigma' in the PF. Higher values = We tolerate more sensor noise.
        self.R = 0.2**2 

        # Each particle is a dictionary containing a Mean (3x1) and Covariance (3x3).
        self.particles = []
        for _ in range(self.num_particles):
            p = {
                'mean': np.array([
                    np.random.uniform(self.x_lim[0], self.x_lim[1]),
                    np.random.uniform(self.y_lim[0], self.y_lim[1]),
                    np.random.uniform(-np.pi, np.pi) 
                ]),
                'cov': np.eye(3) * 0.5, # Initial uncertainty (start with a loose belief)
                'weight': 1.0 / self.num_particles
            }
            self.particles.append(p)

        self.last_odom = None
        self.last_scan = None
        self.ground_truth = None
        self.last_time = self.get_clock().now()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Ground truth used ONLY for calculating error metrics, not for localization
        self.create_subscription(ModelStates, '/gazebo/model_states', self.groundtruth_callback, 10)
        
        self.pub_pose = self.create_publisher(PoseStamped, '/upf_pose', 10)
        self.pub_parts = self.create_publisher(PoseArray, '/upf_particles', 10)
        self.pub_uncertainty = self.create_publisher(Float32, '/upf_uncertainty', 10)
        
        # Update happens at 10Hz
        self.create_timer(0.1, self.update)
        self.get_logger().info("GMM-UKF Initialized")

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

    def get_sigma_points(self, mean, cov):
        """
        Generates 2L+1 Sigma Points that capture the mean and covariance of the distribution.
        This is a crucial step in UKF and is used to deterministically sampling the Gaussian instead of random sampling.
        """
        # 2L+1 Sigma points (row) and L (columns) for x, y, theta
        sigma_points = np.zeros((2 * self.L + 1, self.L))
        # First point is always the mean
        sigma_points[0] = mean

        # Calculate Square Root of Matrix via Cholesky Decomposition
        # S = sqrt((L + lambda) * P)
        scaled_cov = (self.L + self.lamb) * cov
        
        # I have added Regularization (1*e-6) to prevent crash if matrix is singular and add stability
        # This happens if the robot thinks it is perfectly localized, this can also lead to the algorithm blowing up if met with unexpected movement
        # Or movement that may not be as the filter expected
        scaled_cov += np.eye(self.L) * 1e-6
        cholesky_factor = np.linalg.cholesky(scaled_cov)

        # Generate points in + and - directions along the covariance axes
        for sigma_idx in range(self.L):
            sigma_points[sigma_idx + 1] = mean + cholesky_factor[:, sigma_idx]
            sigma_points[sigma_idx + 1 + self.L] = mean - cholesky_factor[:, sigma_idx]
        return sigma_points

    def compute_weights(self):
        """
        Calculates weights for reconstructing the mean and covariance from Sigma points.
        These weights are fixed based on the alpha/beta parameters.
        """
        weights_mean = np.zeros(2 * self.L + 1)
        weights_cov = np.zeros(2 * self.L + 1)

        weights_mean[0] = self.lamb / (self.L + self.lamb)
        weights_cov[0] = weights_mean[0] + (1 - self.alpha**2 + self.beta)

        for sigma_idx in range(1, 2 * self.L + 1):
            weight_value = 1.0 / (2 * (self.L + self.lamb))
            weights_mean[sigma_idx] = weight_value
            weights_cov[sigma_idx] = weight_value
        return weights_mean, weights_cov

    def ukf_predict(self, particle, dt, linear_vel, angular_vel, weights_mean, weights_cov):
        """
        UKF Prediction Step: Propagate the Gaussian forward through the motion model.
        Uses Unscented Transform to handle nonlinear motion without derivatives.
        """
        # Generate Sigma Points from current belief, which is essentially where the model thinks the robot is
        current_sigma_points = self.get_sigma_points(particle['mean'], particle['cov'])
        predicted_sigma_points = np.zeros_like(current_sigma_points)

        # Pass each sigma point through the Motion Model
        for sigma_idx in range(len(current_sigma_points)):
            sigma_x, sigma_y, sigma_theta = current_sigma_points[sigma_idx]

            # Apply motion model (same as Particle Filter, but without noise)
            # UKF handles uncertainty via process noise Q instead
            predicted_sigma_points[sigma_idx, 0], predicted_sigma_points[sigma_idx, 1], predicted_sigma_points[sigma_idx, 2] = apply_velocity_motion_model(
                sigma_x, sigma_y, sigma_theta,
                linear_vel, angular_vel, dt,
                add_noise=False  # UKF uses Q matrix for process noise
            )
            
        # Reconstruct Mean from predicted sigma points
        predicted_mean = np.zeros(self.L)
        predicted_mean[:2] = np.dot(weights_mean, predicted_sigma_points[:, :2])
        
        # I use circular mean for angles because we can't just normally average angles
        # Ex: The average of 359° and 1° should be 0°, not 180°.
        sin_sum = np.dot(weights_mean, np.sin(predicted_sigma_points[:, 2]))
        cos_sum = np.dot(weights_mean, np.cos(predicted_sigma_points[:, 2]))
        predicted_mean[2] = np.arctan2(sin_sum, cos_sum)

        # Reconstruct Covariance from predicted sigma points
        predicted_cov = np.zeros((self.L, self.L))
        for sigma_idx in range(len(current_sigma_points)):
            state_diff = predicted_sigma_points[sigma_idx] - predicted_mean
            state_diff[2] = np.arctan2(np.sin(state_diff[2]), np.cos(state_diff[2]))
            predicted_cov += weights_cov[sigma_idx] * np.outer(state_diff, state_diff)

        # This adds uncertainty because our motion isn't perfect
        predicted_cov += self.Q

        # When I was running my code without this, my error was exploding to millions
        # If the robot is lost in a featureless hallway, P can grow unbounded. Clipping it to the size of the box helped
        # Bound the uncertainty
        if np.trace(predicted_cov) > self.x_lim[1] * self.y_lim[1]:
            predicted_cov *= (self.x_lim[1] * self.y_lim[1] / np.trace(predicted_cov))

        return predicted_mean, predicted_cov

    def ukf_update(self, predicted_mean, predicted_cov, observed_ranges, sensor_max_range, weights_mean, weights_cov):
        """
        UKF Update Step: Incorporate sensor measurements to refine the belief.
        Computes Kalman Gain to by updating the prediction based on what we have observed so far
        """
        # Generate new sigma points from the Predicted belief
        measurement_sigma_points = self.get_sigma_points(predicted_mean, predicted_cov)

        # Transform sigma points through Measurement Model. This uses LiDAR, keeping it consistent with Particle Filter
        predicted_measurement_sigmas = np.zeros((len(measurement_sigma_points), 4))
        for sigma_idx in range(len(measurement_sigma_points)):
            predicted_range = predict_4_ranges(measurement_sigma_points[sigma_idx, 0], measurement_sigma_points[sigma_idx, 1], 
                                               measurement_sigma_points[sigma_idx, 2], self.x_lim, self.y_lim)
            
            # If the particle predicts "Infinity" but the sensor max is 3.5, we must clamp to 3.5
            # otherwise the error calculation will be huge and we'll kill a correct particle (hypotheses).
            predicted_measurement_sigmas[sigma_idx] = np.clip(predicted_range, 0.0, sensor_max_range)

        # Calculate Predicted Measurement Mean (z_hat)
        predicted_measurement_mean = np.dot(weights_mean, predicted_measurement_sigmas)

        # Calculate Innovation Covariance (S) which will basically tell us how much expected error is there
        innovation_cov = np.zeros((4, 4))
        for sigma_idx in range(len(predicted_measurement_sigmas)):
            measurement_diff = predicted_measurement_sigmas[sigma_idx] - predicted_measurement_mean
            innovation_cov += weights_cov[sigma_idx] * np.outer(measurement_diff, measurement_diff)
        innovation_cov += np.eye(4) * self.R
        
        # If S is 0, we can't invert it. Adding epsilon ensures stability. This will usually happen if the sensor sees absolutely nothing
        innovation_cov += np.eye(4) * 1e-6

        # Calculate Cross Covariance (Pxz) to portray Relationship between State and Measurement
        cross_cov = np.zeros((self.L, 4))
        for sigma_idx in range(len(predicted_measurement_sigmas)):
            state_diff = measurement_sigma_points[sigma_idx] - predicted_mean
            state_diff[2] = np.arctan2(np.sin(state_diff[2]), np.cos(state_diff[2]))
            measurement_diff = predicted_measurement_sigmas[sigma_idx] - predicted_measurement_mean
            cross_cov += weights_cov[sigma_idx] * np.outer(state_diff, measurement_diff)

        # Calculate Kalman Gain (K)
        kalman_gain = cross_cov @ np.linalg.pinv(innovation_cov)

        # Innovation: Difference between Real Sensor and Predicted Sensor
        innovation = observed_ranges - predicted_measurement_mean

        # Update State: x = x_pred + K(z - z_hat)
        updated_mean = predicted_mean + kalman_gain @ innovation
        updated_mean[2] = np.arctan2(np.sin(updated_mean[2]), np.cos(updated_mean[2]))

        # Update Covariance: P = P_pred - K S K^T
        updated_cov = predicted_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        updated_cov = (updated_cov + updated_cov.T) / 2.0

        return updated_mean, updated_cov, innovation, innovation_cov

    def ukf_step(self, particle, dt, linear_vel, angular_vel, observed_ranges, sensor_max_range):
        """
        This is the UKF Step for a SINGLE particle (hypothesis).
        Orchestrates the Prediction and Update steps to refine our belief about where the robot is.
        """
        weights_mean, weights_cov = self.compute_weights()
        predicted_mean, predicted_cov = self.ukf_predict(particle, dt, linear_vel, angular_vel, weights_mean, weights_cov)
        updated_mean, updated_cov, innovation, innovation_cov = self.ukf_update(
            predicted_mean, predicted_cov, observed_ranges, sensor_max_range, weights_mean, weights_cov
        )

        # If update throws particle to infinity, I get rid of it.
        if np.linalg.norm(updated_mean[:2]) > 50.0:
             updated_mean = predicted_mean
             updated_cov = predicted_cov
             likelihood = 1e-20
        else:
            likelihood = np.exp(-0.5 * innovation.T @ np.linalg.inv(innovation_cov) @ innovation)

        particle['mean'] = updated_mean
        particle['cov'] = updated_cov
        particle['weight'] *= likelihood

    def get_mahalanobis_dist(self, hypothesis1, hypothesis2):
        """
        Statistical distance between two hypotheses.
        I have used Mahalanobis because it considers Covariance. If two particles 1m apart with huge uncertainty, they should be considered "close".
        """
        mean_diff = hypothesis1['mean'] - hypothesis2['mean']
        mean_diff[2] = np.arctan2(np.sin(mean_diff[2]), np.cos(mean_diff[2]))
        average_cov = (hypothesis1['cov'] + hypothesis2['cov']) / 2.0
        inverse_cov = np.linalg.inv(average_cov)
        distance_squared = mean_diff.T @ inverse_cov @ mean_diff
        return np.sqrt(distance_squared)


    def manage_hypotheses(self):
        """
        Gaussian Mixture Management: Prune, Merge, and Replenish.
        This keeps the number of hypotheses roughly constant while removing weaker ones.
        """
        # 1. Prune: Delete weak hypotheses
        self.particles.sort(key=lambda particle: particle['weight'], reverse=True)
        surviving_particles = []

        # Keep best one always + any others with decent weight
        for i, p in enumerate(self.particles):
            if i == 0 or p['weight'] > 1e-3:
                surviving_particles.append(p)

        # Normalize remaining weights
        total_w = sum(p['weight'] for p in surviving_particles)
        if total_w > 0:
            for p in surviving_particles: p['weight'] /= total_w

        # 2. Merge: Combine duplicate hypotheses
        merged_hypotheses = []
        while len(surviving_particles) > 0:
            primary = surviving_particles.pop(0)
            cluster = [primary]
            remaining = []
            
            for other in surviving_particles:
                mah_dist = self.get_mahalanobis_dist(primary, other)
                if mah_dist < 3.0:
                    cluster.append(other)
                else:
                    remaining.append(other)
            surviving_particles = remaining

            if len(cluster) == 1:
                merged_hypotheses.append(cluster[0])
                continue

            # Merge Logic: Weighted Average of Means & Covariances (Moment Matching)
            w_new = sum(p['weight'] for p in cluster)
            mu_new = np.zeros(3)
            sin_sum = sum(p['weight'] * np.sin(p['mean'][2]) for p in cluster)
            cos_sum = sum(p['weight'] * np.cos(p['mean'][2]) for p in cluster)
            
            for k in range(2):
                mu_new[k] = sum(p['weight'] * p['mean'][k] for p in cluster) / w_new
            mu_new[2] = np.arctan2(sin_sum, cos_sum)
            
            cov_new = np.zeros((3,3))
            for p in cluster:
                diff = p['mean'] - mu_new
                diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))
                cov_new += (p['weight'] / w_new) * (p['cov'] + np.outer(diff, diff))
                
            merged_hypotheses.append({'mean': mu_new, 'cov': cov_new, 'weight': w_new})

        # Instead of cloning the parent, I changed this to spawn completely new random particles. This proved to be specially useful in my symmetric world.
        # This gives the filter a chance to "rediscover" the correct location if it got stuck in the symmetric world.
        min_hypotheses = 4
        while len(merged_hypotheses) < min_hypotheses:
            new_mean = np.array([
                np.random.uniform(self.x_lim[0], self.x_lim[1]),
                np.random.uniform(self.y_lim[0], self.y_lim[1]),
                np.random.uniform(-np.pi, np.pi) 
            ])
            new_cov = np.eye(3) * 0.5
            
            new_particle = {
                'mean': new_mean,
                'cov': new_cov,
                'weight': 1.0 / self.num_particles # Give it a fighting chance
            }
            merged_hypotheses.append(new_particle)

        # Normalize again after adding new particles
        total_final_w = sum(p['weight'] for p in merged_hypotheses)
        if total_final_w > 0:
            for p in merged_hypotheses: p['weight'] /= total_final_w

        self.particles = merged_hypotheses

    def update(self):
        """
        Main Update Loop
        """
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if self.last_odom is None or self.last_scan is None: return
        # If the time jump is too big, skip to keep calculations consistent.
        if dt > 1.0: return

        linear_vel = self.last_odom.twist.twist.linear.x
        angular_vel = self.last_odom.twist.twist.angular.z

        # Don't run update if stationary
        if abs(linear_vel) < 1e-3 and abs(angular_vel) < 1e-3: return

        # Use shared preprocessing logic
        observed_4_ranges, sensor_max_range = preprocess_lidar_observation(self.last_scan)

        # Step 1 : Update Every Hypothesis
        for p in self.particles:
            self.ukf_step(p, dt, linear_vel, angular_vel, observed_4_ranges, sensor_max_range)

        # Step 2 : Normalize Weights so they sum to 1
        total_weight = sum(p['weight'] for p in self.particles)
        for p in self.particles: p['weight'] /= total_weight

        # Step 3 : Manage Hypotheses
        self.manage_hypotheses()

        # Step 4 : Calculate Final Weighted Mean (Estimated Pose)
        final_estimate_mean = np.zeros(3)
        particle_weights = np.array([particle['weight'] for particle in self.particles])
        particle_means = np.array([particle['mean'] for particle in self.particles])

        final_estimate_mean[0] = np.sum(particle_weights * particle_means[:, 0])
        final_estimate_mean[1] = np.sum(particle_weights * particle_means[:, 1])
        sin_sum = np.sum(particle_weights * np.sin(particle_means[:, 2]))
        cos_sum = np.sum(particle_weights * np.cos(particle_means[:, 2]))
        final_estimate_mean[2] = np.arctan2(sin_sum, cos_sum)

        # Calculate spatial spread (weighted std of hypothesis positions)
        # This is comparable to PF's particle spread metric
        weighted_mean_pos = np.average(particle_means[:, :2], weights=particle_weights, axis=0)
        
        # Calculate squared Euclidean distance for each particle (dx^2 + dy^2)
        # This results in a 1D array of shape (N,), which matches the weights shape.
        diffs_squared = (particle_means[:, :2] - weighted_mean_pos)**2
        dists_squared = np.sum(diffs_squared, axis=1) 
        weighted_variance = np.average(dists_squared, weights=particle_weights)
        spatial_spread = np.sqrt(weighted_variance)
        self.pub_uncertainty.publish(Float32(data=float(spatial_spread)))

        # Publish Particle Cloud and Log Error
        if self.ground_truth is not None:
            localization_error = np.linalg.norm(final_estimate_mean[:2] - self.ground_truth)
            num_hypotheses = len(self.particles)
            avg_uncertainty = np.mean([np.linalg.det(particle['cov'][:2,:2]) for particle in self.particles])
            self.get_logger().info(
                f"[UPF] Error: {localization_error:.2f}m | Hypotheses: {num_hypotheses} | Uncertainty: {avg_uncertainty:.4f}"
            )

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.pose.position.x = final_estimate_mean[0]
        pose_msg.pose.position.y = final_estimate_mean[1]
        orientation_quat = Rotation.from_euler('xyz', [0, 0, final_estimate_mean[2]]).as_quat()
        pose_msg.pose.orientation.z = orientation_quat[2]
        pose_msg.pose.orientation.w = orientation_quat[3]
        self.pub_pose.publish(pose_msg)

        if self.pub_parts.get_subscription_count() > 0:
            particle_array_msg = PoseArray()
            particle_array_msg.header = pose_msg.header
            for particle in self.particles:
                particle_pose = Pose()
                particle_pose.position.x = particle['mean'][0]
                particle_pose.position.y = particle['mean'][1]
                particle_quat = Rotation.from_euler('xyz', [0, 0, particle['mean'][2]]).as_quat()
                particle_pose.orientation.z = particle_quat[2]
                particle_pose.orientation.w = particle_quat[3]
                particle_array_msg.poses.append(particle_pose)
            self.pub_parts.publish(particle_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GaussianMixtureUKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()