# TurtleBot3 Localization: Particle Filter vs GMM-UKF

This project implements and compares two localization algorithms for TurtleBot3:
- **Particle Filter (PF)**: Monte Carlo Localization with 1000 particles
- **GMM-UKF**: Gaussian Mixture Model with Unscented Kalman Filter (100 hypotheses)

Both filters use the same motion model from Probabilistic Robotics Table 5.3 and a 4-ray LiDAR measurement model.

NOTE:
The source code has a lot of comments to explain our thought process precisely. For that reason, this README holds the purpose
of helping with the installation and run process of the code. Furthermore, demos can be found under demos/.
---

## Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- TurtleBot3 packages
- Python 3.10+

---

## Installation

### 1. Install ROS 2 Humble and TurtleBot3

If you haven't already, install ROS 2 Humble and TurtleBot3 packages:

```bash
# Install TurtleBot3 packages
sudo apt update
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-teleop

# Set TurtleBot3 model (required for simulation)
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
source ~/.bashrc
```

### 2. Build the Workspace

```bash
# Clone this repo and switch to this branch first
git clone https://github.com/ArqumUddin/prob_rob_labs_ros_2.git
cd /path/to/prob_rob_labs_ros_2
git checkout gmm_ukf_comparison

colcon build --packages-select prob_rob_labs --symlink-install
source install/setup.bash
```

---

## Running the Localization Comparison

You'll need **4 separate terminals** to run the full system (or 5 if you want real-time visualization).

### Terminal 1: Launch TurtleBot3 Simulation

```bash
cd /path/to/prob_rob_labs_ros_2
source install/setup.bash
ros2 launch prob_rob_labs turtlebot3_room_launch.py
```

This launches Gazebo with the TurtleBot3 in a hallway environment with a red box landmark at (-5, 2).

### Terminal 2: Run Particle Filter

```bash
cd /path/to/prob_rob_labs_ros_2
source install/setup.bash
ros2 run prob_rob_labs particle_filter
```

You'll see console output like:
```
[PF] Error: 0.15m | Spread: 2.34m | effective_n_of_particles: 850
```

### Terminal 3: Run GMM-UKF Filter

```bash
cd /path/to/prob_rob_labs_ros_2
source install/setup.bash
ros2 run prob_rob_labs gmm_ukf_filter
```

You'll see console output like:
```
[UPF] Error: 0.12m | Hypotheses: 5 | Uncertainty: 0.0234
```

### Terminal 4: Teleoperate the Robot

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Press `i` to move the robot forward.** The robot will start moving, and both filters will begin localizing.

### Terminal 5 (Optional): Real-time Visualization

```bash
cd /path/to/prob_rob_labs_ros_2
source install/setup.bash
ros2 run prob_rob_labs plot_comparison
```

This will open a matplotlib window with 5 subplots showing:
1. **2D Trajectory** - Ground truth path vs PF and UKF estimates
2. **Localization Error** - Error over time for both filters with mean error lines
3. **X Position vs Time** - X coordinate tracking
4. **Y Position vs Time** - Y coordinate tracking
5. **Uncertainty vs Time** - Spatial spread of belief for both filters
   - **PF**: Standard deviation of particle positions (meters)
   - **UKF**: Weighted standard deviation of hypothesis positions (meters)
   - Both metrics are directly comparable and represent how "spread out" the filter's belief is

The plots update in real-time as the robot moves.

---

## Understanding the Output

### Particle Filter Output
- **Error**: Euclidean distance from ground truth (meters)
- **Spread**: Standard deviation of particle positions (meters) - lower is better
- **effective_n_of_particles**: Effective sample size - measures particle diversity

### GMM-UKF Output
- **Error**: Euclidean distance from ground truth (meters)
- **Hypotheses**: Number of active Gaussian components
- **Uncertainty**: Average covariance determinant - measures filter confidence

### My Analysis and Hypothesis

After analyzing the results, my hypothesis is that while my initial intuition wasn't wrong, it was incomplete. Mathematically, the GMM-UKF is the more sophisticated algorithm, and I expected it to outperform the 'simpler' Particle Filter. However, I underestimated the impact of the environment itself.

My test environment is a long, featureless, symmetric hallway that creates a uniform probability distribution (a flat line of uncertainty). The GMM-UKF struggled here because a Gaussian distribution fundamentally cannot represent a flat line; it forces a peak. This caused the filter to 'guess' a specific location with high confidence, only to jump when the sensors contradicted it, creating the jitter seen in the graphs.

In contrast, the Particle Filter succeeded by brute force. It didn't try to be smart; it simply flooded the hallway with thousands of particles to cover every possibility. This experiment highlighted that while GMM-UKF is mathematically elegant, the 'brute force' of a Particle Filter is often superior in low-dimensional, highly ambiguous environments. The jitter wasn't a bug in the code, but a mismatch between the algorithm's assumptions and the environment's reality.

---

## Visualizing in RViz (Optional)

To visualize the particle clouds and pose estimates:

```bash
rviz2
```

Then add the following topics:
- `/pf_pose` - Particle Filter pose estimate (PoseStamped)
- `/upf_pose` - GMM-UKF pose estimate (PoseStamped)
- `/pf_particles` - Particle cloud visualization (PoseArray)
- `/upf_particles` - GMM-UKF hypothesis visualization (PoseArray)

Demo rviz2 demo video can be found here: https://drive.google.com/file/d/1UtEK-xoE0X3IqPuY-ERmY3FWYKPKjAHY/view?usp=drive_link

---

## Key Features

### Shared Code Architecture
Both filters now share:
- **Motion Model** ([motion_model.py](prob_rob_labs/src/localization/motion_model.py)) - Arc motion from Probabilistic Robotics Table 5.3
- **Ray Tracing** ([ray_tracing.py](prob_rob_labs/src/localization/ray_tracing.py)) - 4-ray LiDAR prediction with landmark detection
- **Measurement Preprocessing** - Consistent sensor data handling

### Filter-Specific Behavior
- **PF**: Adds noise at particle level via `alphas` parameters
- **GMM-UKF**: Adds noise at distribution level via `Q` covariance matrix
- **PF**: Uses simple Gaussian likelihood
- **GMM-UKF**: Uses Mahalanobis distance via innovation covariance

Both filters implement the same physics-based motion model but differ in how they represent uncertainty:

- **PF**: Represents belief as 1000 weighted samples
- **GMM-UKF**: Represents belief as ~4-100 Gaussian distributions

---

## Project Structure

```
prob_rob_labs/
├── src/localization/
│   ├── particle_filter.py      # Monte Carlo Localization
│   ├── gmm_ukf_filter.py        # Gaussian Mixture UKF
│   ├── motion_model.py          # Shared arc motion model
│   ├── ray_tracing.py           # Shared ray tracing utilities
│   └── plot_comparison.py       # Real-time visualization
├── launch/
│   ├── turtlebot3_room_launch.py
│   ├── pf_launch.py
│   └── upf_launch.py
|── worlds/
|   └── room.world               # Hallway environment
└── demos/
    ├── Comparison.png           # Sample Figure
    └── Video                    
```

---
