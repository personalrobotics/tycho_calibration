# Tycho Calibration

This repo contains information to calibrate the Tycho robot (built from HEBI Robotic modules) to prepare for fine manipulation tasks.

This repo contains Python script to calibrates 1) transformation from Optitrack frame to the robot frame and 2) Robot DH links for Forward Kinematics.

Through out the repo, "R" refers to the Optitrack transformation to the robot base frame.

## Walk Through the Complete Calibration Process

To calibrate the robot's kinematic model, you need a motion track system.

### STEP 1. Calibrating the Motion Capture Optitrack Cage

Follow the instruction at [https://github.com/personalrobotics/mocap_optitrack](https://github.com/personalrobotics/mocap_optitrack)

You need to reset the Optitrack ground plane if the table or the cage moved.

You need to recalibrate the cage if the cameras mounted on the cage moved.

### STEP 2. Calibrating the Robot Kinematic Model

Follow the instruction below.

You will estimate (1) the transformation between the Optitrack frame and the robot frame (R), and (2) the DH parameters of all the links on the robot.

To estimate (1), you will command the robot to rotate around its base. You need HIGH-QUALITY recording such that (a) you record the position of one fixed point on the robot while the robot rotates around its base, (b) the robot rotate > 3 complete circles, (c) every joint of the robot holds its position fixed as stable as possible, (d) the Optitrack software can "track" your desired point throughout the process.

To estimate (2), you will command the robot to move and record the joint positions + end-effector positions. You need HIGH-QUALITY recording such that (a) you have a total of 2 min recording, (b) you cover many configurations of the robot, (c) you cover the desired trajectories of our tasks and (d) you cover both static recording and dynamic recording. For static recording you will hold the robot at a fixed location and record a super short rosbag; For dynamic recording, you move the robot using swing/teleoperation.

### STEP 3. Gain Tuning the Robot Controller

Using 'c' to tune the controller gains.

Try tune the gains under swing / step mode, teleop mode.

And tune the velocity controller gains.

### STEP 4. Calibrating the Robot Dynamic Model

Follow the instruction in [tycho_julia](https://github.com/personalrobotics/tycho_julia)。

## In this repo: Calibrate the transformation from Optitrack to the Robot

1. Use tycho_demo 'rotate_base' mode to rotate the robot around the base. By default, you will press 'x' to fix the robot position and 'b' to rotate the robot around the base frame z-axis. If the robot gains are untuned, the robot might fail to hold the rest of the joint positions during the rotation, which will be problematic. Please try to tune the gains or physically support the robot to ensure the rest of joints ARE FIXED during this recording.

2. Record a bag that contains the desired tracking point ('/Ball/point' by default). For this recording, try to place the robot at a good starting position such that (1) the desired point is always tractable and (2) is tracked by the same cameras. Though it is perhaps impossible to achieve both, try to make the reading consistent.

3. Use `src/sync_FK.py` to turn the recording into a CSV file. **As always, before sync ROS messages, ensure to close any ROS node that is running!!!!** Otherwise, the message filter probably will not be able to find any synced message.

4. Use `src/fit_R.py` to fit the transformation using the set of collected data. Remember to adjust both the csv to use and which portion of data to use in the script.

5. Measure the robot base height. Paste into `MEASURED_BASE_HEIGHT` at the top of `fit_R.py`
    - Suggestion for measurement:
      1. Let w be the width of the ball and let h be the height. Then set c=h-w/2, which is about the centroid.
      2. Find 2 mocap balls of the same size. Put them on the T-channel, on either side of the robot base plate.
      3. In Motive, find the heights (y-axis) and average them. Subtract the ground plane marker height. Account for ground plane offset (if nonzero). Subtract c (calculated in 1). Done!

6. Try `python src/fitR.py --csv data/calibration-fitR3.csv --cmaes`. Optionally, NOT passing the cmaes arg to use LBFGS.

7. Paste the result figures and texts into the results folder and commmit.

8. You will need to update:

  - (1) the `calibration.py`, there is a constant named `measured_R`.
  - (2) `tycho_env/utils.py`, the constant named `R_OPTITRACK2BASE`.
  - (3) `tycho_demo/launch/optitrack_transform.sh`. You will need to change the order of the numbers before you fill this file.
  - (4) Any demo launch file should source the script above to publish the transformation between the optitrack and the world. e.g. `tycho_demo`, `tycho_teleop`, `tycho_imitation` etc. If they are not, you need to update the R manually.

## DH Links

1. **REMOVE the residual offsets in tycho_env** (set `OFFSET_JOINTS = np.zeros(7)`) BEFORE you record data for optimizing DH link.

2. Measure the length between the tip of the chopsticks (where you placed the tracking point, the center of the tracking ball) and its holder (the 3D printed part that mounts the chopsticks, the side that is closer to the chopstick tip), fill the number in `src/calibration.py`. The line needs filling has a comment with `<-`.

2. Use tycho_demo to record some movements:

  - Record some teleoperation movements, hoping to collect trajectories similar to the desired trajectories for the task

  - Record some swing movements.

  - Record some static poses.

3. Use sync_FK to turn rosbags to CSV.

4. Measure the DH links and put in `src/calibration.py@line83:measured_FK`. Copy the fit R from above to `measured_R`.

5. Measure the last mile of transformation: from the last end effector to the tip of chopsticks (where the marker is placed).

6. In `src/calibration.py` find the optimization you want to run. Choices are:

- Optimize FK given R; note the cost function punishes deviation from your supplied measurements. You can adjust the weight of this punishment.

- Optimize R and FK iteratively, i.e. fixing one and optimizing the other in turn; You can also use this to optimize R given FK. If you have followed fit_R to obtain the R, you can skip this (and it might be preferred to skip this if you favor having a correct R over having less mismatch between estimated / actual tip locations).

- Optimize R and FK jointly. NOT RECOMMENDED! The problem can be ill-posed.

- Optimize R only. NOT RECOMMENDED! Use Fit_R instead. Using HEBI Robot's factory FK to optimize for R.

6. Usually, we start with Optimize FK given R (Step 2). Optionally, optimize R and FK iteratively.

Examples:

7. In calibration script you can choose which parts of FK params to optimize. Note that the third and fourth of DH parameters (joint offset and theta offset) for joint 1 will help us combat measurement error and drifting-of-zero-point. You should always allow optimize them.

8. The cost function for optimizing FK will punish deviation from the supplied initial values. You can adjust the weight.

9. After you obtain the result:

(1) paste it in the the calibration script

(2) run the script again with STEP 0 to visualize the quality of the transforamtion.

(3) Once done, copy the parameters over to `tycho_env` as FK_params.

(4) use STEP 5 to generate an YAML. Paste to `tycho_description`. Update the URDF model by catkin clean the whole workspace, and catkin build. Once you paste this, Rviz should display the latest robot model.

(5) Find the generated URDF. Copy to `tycho_env`. e.g. `cp devel/share/tycho_description/robots/hebi_chopsticks.urdf src/tycho_env/tycho_env/assets/pybullet.urdf`. This URDF will be used to (1) visualize the robot in PyBullet and (2) generate base XML for Mujoco modeling.


## Training a neural network to predict the backlash

After the above steps, call `python src/train_nn_backlash.py data/your_data.csv` to train a neural network to predict the backlash on each joint.

## Create collision balls on top of the URDF model

Clear cache `catkin clean` then run `catkin build` for this project to generate a new URDF model. Follow the
instructions in `viz_collision_balls.py` to create collision balls.

## Tips

1. When collecting recording for DH links, it is preferred to record static pose. i.e. Place the robot at a fixed location, record, stop record quickly, allow robot to move. This counts as one recording. You need enough recordings to cover the whole config space + you can favor the workstation space by collecting more data there.

2. Please document your FK\_log in results/FK\_log.
