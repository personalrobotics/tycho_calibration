# Hebi Calibration

This scripts calibrates 1) transformation from Optitrack frame to Hebi frame and 2) Hebi DH links for Forward Kinematics.

## Usage

1. Start by measuing the Hebi DH links and put in `src/calibration.py@line83:measured_FK`.

2. Measure the transformation from Optitrack to hebi base frame. 

- Obsolete: ~You can use `src/measure_R.py` to input pairs of Optitrack coordinates and Hebi coordinates. You can read the Optitrack coordinates; but you might have to measure the mocap ball position in hebi frame.~ 

- Preferred: You can use `src/fit_R.py` to fit the transformation using a set of collected data.

3. In `src/calibration.py` find the optimization you want to run. Choices are:

- Optimize FK given R; note the cost function punishes deviation from your supplied measurements. You can adjust the weight of this punishment.

- Optimize R and FK iteratively, i.e. fixing one and optimizing the other in turn; You can also use this to optimize R given FK

- Optimize R and FK jointly. This is not recommended because the problem can be ill-posed.

- Optimize R only (optitrack transformation to hebi base frame) using hebi factory FK

## Tips

1. When collecting recordings, place the robot at a fixed location, record, stop record, allow robot to move. This counts as one recording. You need enough recordings to cover the whole config space + you can favor the workstation space by collecting more data there.

2. Please document your FK\_log in results/FK\_log.

3. Once done, copy the parameters over to `hebi_env`.
