#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from gazebo_msgs.msg import ModelStates
from collections import deque

class FilterComparisonPlotter(Node):
    def __init__(self):
        super().__init__('filter_comparison_plotter')

        # Data storage (keep last 1000 samples)
        self.max_samples = 1000
        self.times = deque(maxlen=self.max_samples)

        # Ground Truth
        self.gt_x = deque(maxlen=self.max_samples)
        self.gt_y = deque(maxlen=self.max_samples)

        # Particle Filter
        self.pf_x = deque(maxlen=self.max_samples)
        self.pf_y = deque(maxlen=self.max_samples)
        self.pf_error = deque(maxlen=self.max_samples)
        self.pf_spread = deque(maxlen=self.max_samples)

        # GMM-UKF
        self.ukf_x = deque(maxlen=self.max_samples)
        self.ukf_y = deque(maxlen=self.max_samples)
        self.ukf_error = deque(maxlen=self.max_samples)
        self.ukf_spread = deque(maxlen=self.max_samples)

        # Current values
        self.current_gt = None
        self.current_pf = None
        self.current_ukf = None
        self.current_pf_uncertainty = None
        self.current_ukf_uncertainty = None
        self.start_time = None

        # Subscriptions
        self.create_subscription(ModelStates, '/gazebo/model_states', self.groundtruth_callback, 10)
        self.create_subscription(PoseStamped, '/pf_pose', self.pf_callback, 10)
        self.create_subscription(PoseStamped, '/upf_pose', self.ukf_callback, 10)
        self.create_subscription(Float32, '/pf_uncertainty', self.pf_uncertainty_callback, 10)
        self.create_subscription(Float32, '/upf_uncertainty', self.ukf_uncertainty_callback, 10)

        # Set up the plot (2 rows, 3 columns)
        self.fig, self.axs = plt.subplots(2, 3, figsize=(20, 10))
        self.fig.suptitle('Particle Filter vs GMM-UKF Comparison', fontsize=16)

        # Animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)

        self.get_logger().info("Filter Comparison Plotter Initialized")
        self.get_logger().info("Waiting for data from filters and ground truth...")

    def groundtruth_callback(self, msg):
        """
        Extract ground truth pose
        """
        robot_idx = msg.name.index('waffle_pi')
        robot_pose = msg.pose[robot_idx]
        self.current_gt = np.array([robot_pose.position.x, robot_pose.position.y])

    def pf_callback(self, msg):
        """
        Store Particle Filter estimate
        """
        self.current_pf = np.array([msg.pose.position.x, msg.pose.position.y])
        self.update_data()

    def ukf_callback(self, msg):
        """
        Store GMM-UKF estimate
        """
        self.current_ukf = np.array([msg.pose.position.x, msg.pose.position.y])
        self.update_data()

    def pf_uncertainty_callback(self, msg):
        """
        Store Particle Filter uncertainty (spatial spread)
        """
        self.current_pf_uncertainty = msg.data

    def ukf_uncertainty_callback(self, msg):
        """
        Store GMM-UKF uncertainty (weighted spatial spread)
        """
        self.current_ukf_uncertainty = msg.data

    def update_data(self):
        """
        Update data arrays when we have all three values
        """
        if self.current_gt is None or self.current_pf is None or self.current_ukf is None:
            return

        if self.start_time is None:
            self.start_time = self.get_clock().now()

        # Calculate elapsed time
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9

        # Store data
        self.times.append(elapsed)

        # Ground truth
        self.gt_x.append(self.current_gt[0])
        self.gt_y.append(self.current_gt[1])

        # Particle Filter
        self.pf_x.append(self.current_pf[0])
        self.pf_y.append(self.current_pf[1])
        self.pf_error.append(np.linalg.norm(self.current_pf - self.current_gt))

        # GMM-UKF
        self.ukf_x.append(self.current_ukf[0])
        self.ukf_y.append(self.current_ukf[1])
        self.ukf_error.append(np.linalg.norm(self.current_ukf - self.current_gt))

        # Store uncertainty values published by the filters
        if self.current_pf_uncertainty is not None:
            self.pf_spread.append(self.current_pf_uncertainty)
        else:
            self.pf_spread.append(0.0)

        if self.current_ukf_uncertainty is not None:
            self.ukf_spread.append(self.current_ukf_uncertainty)
        else:
            self.ukf_spread.append(0.0)

    def update_plot(self, frame):
        """
        Update all subplots
        """
        if len(self.times) < 2:
            return

        # Clear all axes
        for ax in self.axs.flat:
            ax.clear()

        times = np.array(self.times)

        # Plot 1: X-Y Trajectory
        ax1 = self.axs[0, 0]
        ax1.plot(self.gt_x, self.gt_y, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax1.plot(self.pf_x, self.pf_y, 'b--', linewidth=1.5, label='Particle Filter', alpha=0.6)
        ax1.plot(self.ukf_x, self.ukf_y, 'r--', linewidth=1.5, label='GMM-UKF', alpha=0.6)

        # Mark current positions
        if len(self.gt_x) > 0:
            ax1.plot(self.gt_x[-1], self.gt_y[-1], 'ko', markersize=8)
            ax1.plot(self.pf_x[-1], self.pf_y[-1], 'bo', markersize=6)
            ax1.plot(self.ukf_x[-1], self.ukf_y[-1], 'ro', markersize=6)

        ax1.set_xlabel('X Position (m)', fontsize=10)
        ax1.set_ylabel('Y Position (m)', fontsize=10)
        ax1.set_title('2D Trajectory', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Plot 2: Localization Error Over Time
        ax2 = self.axs[0, 1]
        ax2.plot(times, self.pf_error, 'b-', linewidth=1.5, label='Particle Filter Error', alpha=0.7)
        ax2.plot(times, self.ukf_error, 'r-', linewidth=1.5, label='GMM-UKF Error', alpha=0.7)

        # Calculate and display mean errors
        if len(self.pf_error) > 10:
            pf_mean = np.mean(list(self.pf_error)[-100:])  # Last 100 samples
            ukf_mean = np.mean(list(self.ukf_error)[-100:])
            ax2.axhline(pf_mean, color='b', linestyle=':', alpha=0.5, label=f'PF Mean: {pf_mean:.3f}m')
            ax2.axhline(ukf_mean, color='r', linestyle=':', alpha=0.5, label=f'UKF Mean: {ukf_mean:.3f}m')

        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Localization Error (m)', fontsize=10)
        ax2.set_title('Error vs Time', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: X Position Over Time
        ax3 = self.axs[1, 0]
        ax3.plot(times, self.gt_x, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax3.plot(times, self.pf_x, 'b--', linewidth=1.5, label='Particle Filter', alpha=0.6)
        ax3.plot(times, self.ukf_x, 'r--', linewidth=1.5, label='GMM-UKF', alpha=0.6)

        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('X Position (m)', fontsize=10)
        ax3.set_title('X Position vs Time', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Y Position Over Time
        ax4 = self.axs[1, 1]
        ax4.plot(times, self.gt_y, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax4.plot(times, self.pf_y, 'b--', linewidth=1.5, label='Particle Filter', alpha=0.6)
        ax4.plot(times, self.ukf_y, 'r--', linewidth=1.5, label='GMM-UKF', alpha=0.6)

        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Y Position (m)', fontsize=10)
        ax4.set_title('Y Position vs Time', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # Plot 5: Uncertainty/Spread Comparison
        ax5 = self.axs[0, 2]
        ax5.plot(times, self.pf_spread, 'b-', linewidth=1.5, label='PF Spread', alpha=0.7)
        ax5.plot(times, self.ukf_spread, 'r-', linewidth=1.5, label='UKF Spread', alpha=0.7)

        # Calculate and display mean spread
        if len(self.pf_spread) > 10:
            pf_mean_spread = np.mean(list(self.pf_spread)[-100:])
            ukf_mean_spread = np.mean(list(self.ukf_spread)[-100:])
            ax5.axhline(pf_mean_spread, color='b', linestyle=':', alpha=0.5, label=f'PF Mean: {pf_mean_spread:.3f}m')
            ax5.axhline(ukf_mean_spread, color='r', linestyle=':', alpha=0.5, label=f'UKF Mean: {ukf_mean_spread:.3f}m')

        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.set_ylabel('Belief Spread (m)', fontsize=10)
        ax5.set_title('Uncertainty vs Time', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

def main(args=None):
    rclpy.init(args=args)
    node = FilterComparisonPlotter()
    plt.ion()
    plt.show()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()
