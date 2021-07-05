# Tycho Calibration

This scripts calibrates 1) transformation from Optitrack frame to the robot frame and 2) Robot DH links for Forward Kinematics.

Throughout the doc, R = Optitrack transformation to the robot base frame.

##  Transform from Optitrack to the Robot

1. Use tycho_demo 'rotate_base' mode to rotate the robot around the base. By default, you will press 'x' to fix the robot position and 'b' to rotate the robot around the base frame z-axis.

2. Record a bag that contains the desired tracking point ('/Ball/point' by default). For this recording, try to place the robot at a good starting position such that (1) the desired point is always tractable and (2) is tracked by the same cameras. Though it is perhaps impossible to achieve both, try to make the reading consistent.

3. Use `src/sync_FK.py` to turn the recording into a CSV file. **As always, before sync ROS messages, ensure to close any ROS node that is running!!!!** Otherwise, the message filter probably will not be able to find any synced message.

3. Use `src/fit_R.py` to fit the transformation using the set of collected data. Remember to adjust both the csv to use and which portion of data to use in the script.

4. Try `python src/fitR.py --csv data/calibration-fitR3.csv --cmaes`. Optionally, NOT passing the cmaes arg to use LBFGS.

4. Paste the result figures and texts into the results folder and commmit.

5. You will need to update: (1) the `calibration.py` (2) `tycho_env/utils.py` and (3) any demo launch file that publishes the transformation between the optitrack and .

## DH Links

1. First, **REMOVE the residual offsets in tycho_env** (set to zero). Then use tycho_teleop's teleoperation mode to record some movements. Additionally, record some static poses.

2. Use sync_FK to turn to CSV.

3. Measure the DH links and put in `src/calibration.py@line83:measured_FK`. Copy the fit R from above to `measured_R`. The height estimated can be off, therefore we supply a measured number. This number will be fine tuned in the optimization process.

4. Measure the last mile of transformation: from the last end effector to the tip of chopsticks (where the marker is placed).

5. In `src/calibration.py` find the optimization you want to run. Choices are:

- Optimize FK given R; note the cost function punishes deviation from your supplied measurements. You can adjust the weight of this punishment.

- Optimize R and FK iteratively, i.e. fixing one and optimizing the other in turn; You can also use this to optimize R given FK. If you have followed fit_R to obtain the R, you can skip this (and it might be preferred to skip this if you favor having a correct R over having less mismatch between estimated / actual tip locations).

- Optimize R and FK jointly. NOT RECOMMENDED! The problem can be ill-posed.

- Optimize R only. NOT RECOMMENDED! Use Fit_R instead. Using HEBI Robot's factory FK to optimize for R.

6. Usually, we start with Optimize FK given R (Step 2). Optionally, optimize R and FK iteratively.

7. In calibration script you can choose which parts of FK params to optimize. Note that the third and fourth of DH parameters (joint offset and theta offset) for joint 1 will help us combat measurement error and drifting-of-zero-point. You should always allow optimize them.

8. The cost function for optimizing FK will punish deviation from the supplied initial values. You can adjust the weight.

9. After you obtain the result:

(1) paste it in the the calibration script

(2) run the script again with STEP 0 to visualize the quality of the transforamtion.

(3) Once done, copy the parameters over to `tycho_env`.

(4) use STEP 5 to generate an YAML. Paste to `tycho_description`. Update the URDF model by catkin clean and catkin build.


## Tips

1. When collecting recording for DH links, it is preferred to record static pose. i.e. Place the robot at a fixed location, record, stop record quickly, allow robot to move. This counts as one recording. You need enough recordings to cover the whole config space + you can favor the workstation space by collecting more data there.

2. Please document your FK\_log in results/FK\_log.