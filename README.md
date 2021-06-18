# Hebi Calibration

This scripts calibrates 1) transformation from Optitrack frame to Hebi frame and 2) Hebi DH links for Forward Kinematics.

Throughout the doc, R = Optitrack transformation to hebi base frame.

##  Transform from Optitrack to Hebi

1. Use hebi_teleop 'rotate_base' mode (pressing 'b' by default) to rotate the robot around the base.

2. Record a bag that contains ONE desired point ('/Ball/point' by default). During recording, try to place the robot at a good starting position such that (1) the desired point is always tractable and (2) is tracked by the same cameras. Though it is perhaps impossible to achieve both, try to make the reading consistent.

3. Use `src/sync_FK.py` to turn the recording into a CSV file. (As always, before sync ROS messages, ensure to close any ROS nodes that are running.)

3. Use `src/fit_R.py` to fit the transformation using the set of collected data. Remember to adjust both the csv to use and which portion of data to use in the script.

4. Try `python src/fitR.py --csv data/calibration-fitR3.csv --cmaes`. Optionally not using the cmaes arg will use LBFGS.

4. Paste the result figures and texts into the results folder and commmit.

5. You will need to update: (1) the calibration script (2) hebi_env utils and (3) the teleop launch file.

## Hebi DH Links

1. Use hebi_teleop teleoperation mode to record some movements. Preferably, record static poses.

2. Use sync_FK to turn to CSV.

3. Measure the Hebi DH links and put in `src/calibration.py@line83:measured_FK`. Copy the fit R from above to `measured_R`. The height estimated can be off, for now we use a measured number from a while ago.

4. Measure the last mile of transformation: from the last end effector to the tip of chopsticks (where the marker is placed).

5. In `src/calibration.py` find the optimization you want to run. Choices are:

- Optimize FK given R; note the cost function punishes deviation from your supplied measurements. You can adjust the weight of this punishment.

- Optimize R and FK iteratively, i.e. fixing one and optimizing the other in turn; You can also use this to optimize R given FK. If you have followed fit_R to obtain the R, you can skip this (and it might be preferred to skip this if you favor having a correct R over having less mismatch between estimated / actual tip locations).

- Optimize R and FK jointly. NOT RECOMMENDED! The problem can be ill-posed.

- Optimize R only. NOT RECOMMENDED! Use Fit_R instead. Using hebi factory FK to optimize for R.

6. Usually, we start with Optimize FK given R. And optionally optimize R and FK iteratively.

7. In calibration script you can choose which parts of FK params to optimize. Note that the third and fourth of DH parameters (joint offset and theta offset) for joint 1 will help us combat measurement error and drifting-of-zero-point. You should always allow optimize them.

8. The cost function for optimizing FK will punish deviation from the supplied initial values. You can adjust the weight.

9. After you obtain the result, paste it in the the calibration script and run the script again with STEP 0 to visualize the quality of the transforamtion.


## Tips

1. When collecting recording for DH links, it is preferred to record static pose. i.e. Place the robot at a fixed location, record, stop record, allow robot to move. This counts as one recording. You need enough recordings to cover the whole config space + you can favor the workstation space by collecting more data there.

2. Please document your FK\_log in results/FK\_log.

3. Once done, copy the parameters over to `hebi_env`.
