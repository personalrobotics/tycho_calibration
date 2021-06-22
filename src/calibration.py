from __future__ import print_function

import os
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.transform import Rotation as scipyR
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import cm
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================
# Pre-set and initial params
# ==============================================================

measured_R = np.array([0, 0, 0, 1, # quat x y z w, almost identity
     0,0,0])

# 2021.06.21 from fit_R.py
measured_R = np.array([0.00714843, 0.00664545, -0.00000020, 0.99995237,
  -1.07350576, 0.17235089, 2.43135210])
measured_R[-1] = -0.02163 + 0.0162 # Some measured data from a while ago

measured_FK = np.array([
     # link twist (alpha); link length (a);  joint offset (d); theta_offset;
     0,       0,        0.101,  0, # 0  x  2  3
     np.pi/2, 0,        0.0826, 0, # 4  x  6  7
     np.pi,   0.3255,   0.0451, 0, # 8  9  10 11
     np.pi,   0.3255,   0.0713, 0, # 12 13 14 15
     np.pi/2, 0,        0.1143, 0, # 16 x  18 19
     np.pi/2, 0,        0.1143, 0, # 20 x  22 23
     -0.707,  0,   0, 0.707,       # x  x  x  x
     # from end to DH to tip:      # x x x x 28 29 30
     0.101,                        # <- check below for more info
     0.0803,                       # Expect to not change
     0.025])                       # Expect to not change
measured_FK[-3] = (measured_FK[-3]
  + 0.0095/2  # half the mount
  + 0.014908) # from center of the robot hole to the center of the mount screw hole

# Note the DH transformation for the 6 joints moves to the intersection of wirst4 and chop joint
# this is a virtual point that is NOT on the robot!!
# To adjust the last mile of transformation, one usually just need to measure the distance
# between the side of the mount (closer to the tip) to the center of the tracking ball
# and put the number on the line marked with <-
# It is probably better to fine tune this number as well in the optimization process

# You can paste your optimized FK below
optimized_FK = np.array([ # 2021 June 21
 -0.00046457626085699363, 0.00000000000000000000, 0.10150211960557554980, -0.02190870190122657865,
 1.57079115315484107995, 0.00000000000000000000, 0.08220017515303833233, -0.00889059579215141678,
 3.14304091575809030346, 0.32624426386008442691, 0.04544803470806729057, 0.00499185515808455750,
 3.13525427400942380274, 0.32755018637621818867, 0.07096225119083582333, -0.00677021524394086274,
 1.57079620883963033684, 0.00000000000000000000, 0.11430138826329204471, -0.00040271304424944560,
 1.57054043416849231640, 0.00000000000000000000, 0.11430111140998718000, -0.00000131560317569111,
 -0.70699999999999996181, 0.00000000000000000000, 0.00000000000000000000, 0.70699999999999996181,
 0.11932266562500976059, 0.07968843088402800812, 0.02500111140960649056, ])

# ==============================================================
# Utilities for cost function
# ==============================================================

def get_DH_transformation(alpha,a,_theta,d,theta_offset=0):
  theta = _theta + theta_offset
  # Given the DH link construct the transformation matrix
  rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha)],
                 [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha)]])

  trans = np.array([a,-d*np.sin(alpha),np.cos(alpha)*d]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  m = np.vstack((np.hstack((rot, trans)),last_row))
  return m

def get_transformation_matrix(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_dcm() # scipy >=1.4.0 will always normalize quat
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def calculate_FK_transformation(FKparams, joint_position):
  # Given a list of FKparams, shape N by 3, return transformation
  ee = np.eye(4)
  for (alpha, a, d, offset), theta in zip(FKparams, joint_position):
    ee = ee.dot(get_DH_transformation(alpha, a, theta, d, offset))
  return ee

def get_hebi_fk(joint_positions):
  from hebi_env.arm_container import create_empty_robot_default_hrdf
  arm = create_empty_robot_default_hrdf()
  return np.array([np.array(arm.get_FK_ee(p)) for p in joint_positions]) # data_size x 4 x 4

def get_hebi_fk_tips(list_of_hebiee):
  tips = []
  for hebiee in list_of_hebiee:
    x_axis,y_axis,z_axis = hebiee[0:3,0:3].T
    init_position = np.array(hebiee[0:3,3]).reshape(3)
    # update the orignial point from hole to the axis of bottom chop
    position = (init_position + 0 * x_axis +
                 (0.007 - 0.0001) * y_axis + # 0.007 - 0.0001 is about the shift from holder screw to center of chop
                 (0.0017 + 0.0065) * z_axis) # axis coming out from the module plate, 0.0017 holder offset to module plate, 0.0065 bring to chopsticks center
    position = position + x_axis * (0.0035+0.11) # Tip of bottom chopsticks on robot, 0.0035 is half of the holder width, 0.1135 is the first part lengh of the chopsticks
    tips.append(position)
  return tips

def get_ball_in_hebi_frame(list_ball, R_params):
  R = get_transformation_matrix(R_params)
  list_ball_in_hebi_frame = []
  _ball = np.ones(4)
  for ball in list_ball:
    _ball[0:3] = ball
    list_ball_in_hebi_frame.append(R.dot(_ball)[0:3].reshape(3))
  return list_ball_in_hebi_frame

def get_fk_tips(list_jp, FK_params):
  DH_params = np.reshape(FK_params[:24], (6,4)) # each link is represented by 4 params
  last_transformation = get_transformation_matrix(FK_params[-7:])
  list_fk_tips = []
  for jp in list_jp:
    ee = calculate_FK_transformation(DH_params, jp)
    ee = ee.dot(last_transformation)
    list_fk_tips.append(ee[0:3, 3])
  return np.array(list_fk_tips).reshape(-1,3)

# ==============================================================
# Optimization cost and initial params
# ==============================================================

def optimize_R_using_hebi_FK(list_ball, list_tip, initP=None):
  if initP is None:
    initP = np.array(measured_R)

  def cost_func(p, verbose=False):
    loss = []
    p[0:4] = measured_R[0:4] # FIX THE R ROTATION
    R = get_transformation_matrix(p)
    _ball = np.ones(4)
    for ball, hebi_tip in zip(list_ball, list_tip):
      _ball[0:3] = ball
      transform = R.dot(_ball)[0:3]
      loss.append(np.linalg.norm((transform - hebi_tip).reshape(3))) # Euclidean norm
    return np.average(loss) if not verbose else loss
  return initP, cost_func

def optimize_FK_and_R(initRparam, initFKparam, list_ball, list_jp):
  initP = np.hstack((initRparam, initFKparam)).reshape(-1)
  def cost_func(p, verbose=True):
    loss = []
    R_params = np.reshape(p[:7], -1)
    R = get_transformation_matrix(R_params)
    pad_ball = np.ones((len(list_ball),4))
    pad_ball[:,0:3] = np.array(list_ball)
    DH_params = np.reshape(p[7:25], (6,3))
    last_transformation = get_transformation_matrix(p[-7:])
    for ball, cp in zip(pad_ball, list_jp):
      ee = calculate_FK_transformation(DH_params, cp)
      ee = ee.dot(last_transformation)
      prediction = ee[0:3, 3].reshape(3)
      loss.append(np.linalg.norm(R.dot(ball)[0:3] - prediction))
    return np.average(loss) if not verbose else loss
  return initP, cost_func

# --------------------------------------------------
# Parallel optimization of FK

from multiprocessing import Pool
from functools import partial

NUM_POOL = 8

def FK_cost_fn_parallel(DH_params, last_transformation, list_ball_in_hebi_frame, list_jp, indexes):
  loss = []
  for idx in indexes:
    ball = list_ball_in_hebi_frame[idx]
    cp = list_jp[idx]
    ee = calculate_FK_transformation(DH_params, cp)
    ee = ee.dot(last_transformation)
    prediction = ee[0:3, 3].reshape(3)
    loss.append(np.linalg.norm(prediction - ball))
  return loss

pool = Pool(NUM_POOL)

def not_outlier(arr, allowable_deviation=3):
  mean = np.mean(arr)
  std = np.std(arr)
  dis = abs(arr - mean)
  not_outlier =  dis < allowable_deviation * std
  return not_outlier

def optimize_FK_only_parallel(list_ball_in_hebi_frame, list_jp, initP=None, sel_params=np.arange(31)):
  if initP is not None:
    defaultP = np.array(initP)
  else:
    defaultP = np.array(measured_FK)
  initP = defaultP[sel_params]

  def cost_func(_p, verbose=False):
    loss = []
    p = np.array(defaultP)
    p[sel_params] = _p
    #p[8] += p[11] - measured_FK[11] + p[5] - measured_FK[5] ####### Consider add this constraint?
    DH_params = p[:24].reshape(6,4)
    last_transformation = get_transformation_matrix(p[-7:])

    # divide the workload
    my_func = partial(FK_cost_fn_parallel, DH_params, last_transformation, list_ball_in_hebi_frame, list_jp)
    n = len(list_ball_in_hebi_frame)
    n = (n//NUM_POOL) * NUM_POOL
    indexes = np.arange(n).reshape(NUM_POOL,-1)
    pool_results = pool.map(my_func, indexes)
    loss = np.array(sum(pool_results, []))
    loss = loss[not_outlier(loss)]

    # punish the deviation
    deviation_loss = np.exp(np.abs(p - measured_FK) * 10) - 1
    deviation_loss[2] = 0 # don't punish the joint offset on the base which determines the height
    deviation_loss[3] = 0 # don't punish the theta offset on the base to allow rotation around z-axis
    deviation_loss = 0.025 * np.sum(deviation_loss) / (len(sel_params) - 2)
    print('avg_cost {:.20f} deviation loss  {:.10f}'.format(np.average(loss), deviation_loss))
    return np.average(loss) + deviation_loss if not verbose else loss
  return initP, cost_func

# ==============================================================
# Optimizer
# ==============================================================

def cmaes(func, initP, var=1):
  import cma
  es = cma.CMAEvolutionStrategy(initP, var)
  best_so_far = func(initP)
  best_params = initP
  while not es.stop():
    solutions = es.ask()
    f_vals = [func(s) for s in solutions]
    es.tell(solutions, f_vals)
    if np.min(f_vals) < best_so_far:
      best_so_far = np.min(f_vals)
      best_params = solutions[np.argmin(f_vals)]
      print('CMAES found a new set of best params, achieving', best_so_far)
      print('params', best_params)
    es.logger.add()
    es.disp()
  es.result_pretty()
  return best_params

def scipy_optimize(func, initP, method='BFGS', max_func=15000, iprint=1, save=None):
  # Run scipy optimization
  res = scipy.optimize.minimize(func, initP, method=method, options={'disp': None, 'maxfun': max_func, 'iprint': iprint})
  # For more options see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
  print('After optimize, minimum=', func(res.x))
  print("Scipy optimized params", res.x)
  (save and np.savetxt('results/'+save, res.x, delimiter=',',fmt='%f'))
  return res

import argparse
def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="data/ball_and_jointpos.csv")
    parser.add_argument('--max_func', type=int, default=10000)
    parser.add_argument('-s','--step', type=int, default=0)
    parser.add_argument('-f','--filter', type=float, default=0.)
    return parser

if __name__ == '__main__':
  # ===========================================================================
  # Preparation
  # ---------------------------------------------------------------------------
  args = construct_parser().parse_args()

  # Make results folder
  from datetime import datetime
  foldername = str(datetime.now().strftime("calib-%m%d-%H-%M-%S")) + "-STEP{}".format(args.step)
  os.mkdir(foldername)
  f = open(os.path.join(foldername,'calibration.txt'),'a')
  print("Using CSV {}".format(args.csv), file=f)

  # Load data from CSV that contains ball (optitrack tip location) and jp (joint positions)
  df = pd.read_csv(args.csv)
  list_ball = [np.fromstring(r[1:-1], dtype=np.float, sep=' ') for r in df['ball_loc'].to_list()] #[1:-1] to exclude '['']'
  list_jp = [np.fromstring(r[1:-1], dtype=np.float, sep=' ')[0:6]  for r in df['joint_position'].to_list()] # keep only 6 joints
  print("size of datapoints:", len(list_ball))
  print("first ball", list_ball[0])
  print("first jp", list_jp[0])

  # initialize the parameters to use the default dummy params
  R_params, _ = optimize_R_using_hebi_FK(None, None)
  FK_params, _ = optimize_FK_only_parallel(None, None)

  # ===========================================================================
  # Filter
  # ---------------------------------------------------------------------------
  if args.filter > 0:
    print("Removing data points that differ from the existing FK")
    print("Filter threshold:", args.filter)
    print("Filter threshold:", args.filter, file=f)
    fk_tips = get_fk_tips(list_jp, FK_params)
    track_tips = get_ball_in_hebi_frame(list_ball, R_params)
    diff = fk_tips - track_tips
    diff_norm = np.linalg.norm(diff, axis=1)
    val_idx = diff_norm < args.filter
    old_datapoint_count = len(list_ball)
    list_jp = np.array(list_jp)[val_idx]
    list_ball = np.array(list_ball)[val_idx]
    new_datapoint_count = len(list_ball)
    print("Removing {} out of {} datapoints".format(
        old_datapoint_count - new_datapoint_count,
        old_datapoint_count))
    print("Removing {} out of {} datapoints".format(
        old_datapoint_count - new_datapoint_count,
        old_datapoint_count), file=f)

  # Extract Hebi Default FK EE
  list_hebiee = get_hebi_fk(list_jp)
  # expecting hebiee to be at where the chopstick holder touch the bottom plate, should be defined in arm_container
  list_hebiee_tip = get_hebi_fk_tips(list_hebiee)
  print("first Hebi EE\n", list_hebiee[0])
  print("first Hebi calculated tip\n", list_hebiee_tip[0])

  initP, cost_func = optimize_R_using_hebi_FK(list_ball, list_hebiee_tip)
  init_distance = cost_func(initP, verbose=True)
  print('Using HEBI default FK and our measured R...')
  print('Before optimize, avg distance =', np.average(init_distance))
  print('Before optimize, max distance = ', np.max(init_distance))
  print('Before optimize, the worst datapoint is ', list_ball[np.argmax(init_distance)], list_hebiee_tip[np.argmax(init_distance)])

  # check the accuracy of some optimized FK (if exists)
  if args.step == 0 and optimized_FK is not None:
      print('Using parameters:', optimized_FK, file=f)
      list_ball_in_hebi_frame = get_ball_in_hebi_frame(list_ball, R_params)
      initP, cost_func = optimize_FK_only_parallel(list_ball_in_hebi_frame, list_jp, initP=optimized_FK, sel_params=[0])
      initLoss = cost_func(initP, verbose=True)
      print('The supplied optimized_FK has avg distance =', np.average(initLoss), file=f)
      print('The supplied optimized_FK has max distance = ', np.max(initLoss), file=f)

      sns.distplot(initLoss)
      plt.savefig(os.path.join(foldername, 'costs-dist.jpg'))
      x = np.arange(len(initLoss))
      sns.jointplot(x=x, y=initLoss)
      plt.savefig(os.path.join(foldername, 'costs-seq.jpg'))

      arr = np.array(list_ball[:len(initLoss)])
      x = arr[:, 0]
      y = arr[:, 1]
      z = arr[:, 2]
      fig = plt.figure(figsize=(8,6))
      ax = fig.add_subplot(111,projection='3d')
      n = 100
      cmap = cm.RdGy
      colors = cmap(initLoss/max(initLoss))
      colmap = cm.ScalarMappable(cmap=cmap)
      colmap.set_array(initLoss)
      yg = ax.scatter(x, y, z, c=colors, marker='o')
      cb = fig.colorbar(colmap)
      plt.savefig(os.path.join(foldername, 'viz_errors.jpg'))
      plt.show()

  np.set_printoptions(suppress=True, formatter={'float_kind':'{:.20f},'.format}, linewidth=100) # Try to print 4 numbers on one line


  # ----------------------------------------------------------------------------
  # STEP1: Optimize R
  # ----------------------------------------------------------------------------
  if args.step == 1:
    print("\n\nOptimize the transformation matrix R from optitrack frame to hebi\n\n")
    # scipy optimize
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=args.max_func, iprint=args.max_func//50).x
    est_R = res
    est_R[0:4] = np.array(est_R[0:4]) / np.linalg.norm(est_R[0:4]) # normalize quat
    print("Estimated R from optitrack to base", est_R)
    print("Compared with initial P:", initP)
    newCost = cost_func(res, verbose=True)
    print('After optimize, avg distance =', np.average(newCost))
    print('After optimize, max distance = ', np.max(newCost))
    # cmaes optimize
    # res = cmaes(cost_func, initP)
    # res[0:4] = np.array(res[0:4]) / np.linalg.norm(res)
    # print("CMEAS (perhaps more of a global optim)", res)
    R_params = res

  # ----------------------------------------------------------------------------
  # STEP2: Optimize Hebi FK
  # ----------------------------------------------------------------------------
  if args.step == 2:
    print("\n\nOptimize FK function whihe fixing R\n\n")

    def opt_fk(sel_params):
      print("Optimizing select parameters for FK, sel:", sel_params)
      print("Optimizing select parameters for FK, sel:", sel_params, file=f)

      list_ball_in_hebi_frame = get_ball_in_hebi_frame(list_ball, R_params)
      initP, cost_func = optimize_FK_only_parallel(list_ball_in_hebi_frame, list_jp, initP=FK_params, sel_params=sel_params)

      initLoss = cost_func(initP, verbose=True)
      print('Before optimize, avg distance =', np.average(initLoss), file=f)
      print('Before optimize, max distance = ', np.max(initLoss), file=f)
      res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=args.max_func, iprint=args.max_func//50).x
      newCost = cost_func(res, verbose=True)
      print('After optimize, avg distance =', np.average(newCost), file=f)
      print('After optimize, max distance = ', np.max(newCost), file=f)

      return res, newCost

    default_sel_params = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 28, 29, 30]
    optimize_orders = [default_sel_params]

    for opt_params in optimize_orders:
      new_FK_params, newCost = opt_fk(opt_params)
      FK_params[opt_params] = new_FK_params
      ##force the p[8] follow such rule if you add the constraint in cost fn
      #FK_params[8] += FK_params[5] - measured_FK[5] + FK_params[11] - measured_FK[11]


      print("Completed one round of optimization.")
      print("Here are the new FK values:")
      print(FK_params)

      print(FK_params, file=f)
      print("Changes from measured_FK", file=f)
      print(FK_params - measured_FK, file=f)

      sns.distplot(newCost)
      plt.savefig(os.path.join(foldername, 'newCost-dist.jpg'))
      x = np.arange(len(newCost))
      sns.jointplot(x=x, y=newCost)
      plt.savefig(os.path.join(foldername, 'newCost-seq.jpg'))

    f.close()

  FK_params = optimized_FK

  # ----------------------------------------------------------------------------
  # STEP3: Optimize R and FK iteratively
  # ----------------------------------------------------------------------------
  if args.step == 3:
      FK_params = optimized_FK
      print("\n\nOptimize R and FK iteratively")
      initP, cost_func = optimize_FK_and_R(R_params, FK_params, list_ball, list_jp)
      initCost = cost_func(initP, verbose=True)
      print("Before optimization, avg distance", np.average(initCost))
      print("Max distance", np.max(initCost))
      for _ in range(1):
        list_fk_tips = get_fk_tips(list_jp, FK_params)
        _, cost_func = optimize_R_using_hebi_FK(list_ball, list_fk_tips)
        R_params = scipy_optimize(cost_func, R_params, method='L-BFGS-B', max_func=args.max_func, iprint=args.max_func//50).x
        R_params[0:4] = R_params[:4] / np.linalg.norm(R_params[:4])
        print('New R params', R_params)
        newCost = cost_func(R_params, verbose=True)
        print("New average distance", np.average(newCost))
        print("Max distance", np.max(newCost))
        print('\n\n')
        exit()
        list_ball_in_hebi_frame = get_ball_in_hebi_frame(list_ball, R_params)
        _, cost_func = optimize_FK_only(list_ball_in_hebi_frame, list_jp)
        FK_params = scipy_optimize(cost_func, FK_params, method='L-BFGS-B', max_func=args.max_func, iprint=args.max_func//50).x
        newP, cost_func = optimize_FK_and_R(R_params, FK_params, list_ball, list_jp)
        newCost = cost_func(newP, verbose=True)
        print("New average distance", np.average(newCost))
        print("Max distance", np.max(newCost))

  # ----------------------------------------------------------------------------
  # ??? STEP4: Optimize R and FK jointly
  # ----------------------------------------------------------------------------
  if args.step == 4:
    print("\n\nJointly optimize R and FK\n\n")
    initP, cost_func = optimize_FK_and_R(R_params, FK_params, list_ball, list_jp)
    print('Before optimize, avg distance =', cost_func(initP))
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=30000, iprint=50).x
    print('Optimized distance', cost_func(res))

  if args.step == 5:
    import yaml
    file_path = os.path.join(foldername, 'kinematic_parameters.yaml')
    yaml_dict = {
      'fk': optimized_FK.flatten().tolist(),
      'alpha': optimized_FK[[0,4,8,12,16,20]].tolist(),
      'a': optimized_FK[[1,5,9,13,17,21]].tolist(),
      'd': optimized_FK[[2,6,10,14,18,22]].tolist(),
      'trans': optimized_FK[[28, 29, 30]].tolist(),
    }
    with open(file_path, 'w') as yaml_file:
      documents = yaml.dump(yaml_dict, yaml_file)
    print("Wrote parameters to the yaml file")