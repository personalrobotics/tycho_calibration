from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.transform import Rotation as scipyR
from functools import partial

# ==============================================================
# Utilities for cost function
# ==============================================================

def get_DH_transformation(alpha,a,theta,d):
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
  for (alpha, a, d), theta in zip(FKparams, joint_position):
    ee = ee.dot(get_DH_transformation(alpha, a, theta, d))
  return ee

def get_hebi_fk(joint_positions,
                arm_hrdf):
  from hebi_env.arm_container import create_empty_robot
  arm = create_empty_robot(arm_hrdf)
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
    position = position + x_axis * (0.0035+0.112) # Tip of bottom chopsticks on robot, 0.0035 is half of the holder width, 0.11 is the first part lengh of the chopsticks
    tips.append(position)
  return tips

# ==============================================================
# Optimization cost and initial params
# ==============================================================

def optimize_R_using_hebi_FK(list_m6, list_hebiee_tip, initP=None):
  initP = initP or np.array(
    # optimized result from 20200706
    [-0.00750097, 0.01674101, 0.00895461, 0.99979162, # quat x y z w, almost identity
     -1.4426,-0.5949,0.0058] # x y z shift
  )
  def cost_func(p, verbose=False):
    loss = []
    R = get_transformation_matrix(p)
    _m6 = np.ones(4)
    for m6, hebi_tip in zip(list_m6, list_hebiee_tip):
      _m6[0:3] = m6
      transform = R.dot(_m6)[0:3]
      loss.append(np.linalg.norm((transform - hebi_tip).reshape(3))) # Euclidean norm
    return np.average(loss) if not verbose else loss
  return initP, cost_func

def optimize_FK_only(list_m6_in_hebi_frame, list_jp, initP=None):
  initP = initP or np.array([
    #0,0,0.11,
    #np.pi/2,0,0.11,
    #np.pi,0.3,0.09,
    #np.pi,0.3,0.1,
    #np.pi/2,0,0.11,
    #np.pi/2,0,0.11,
    #-0.707, 0, 0, 0.707, 0.13, 0.08, 0.03 # from end to DH to tip
    #])
  # optimized on 20200706
   -7.54514008e-03, 2.45281047e-02, 7.08143884e-02,
    1.56589303e+00, 1.33388682e-02, 1.03374056e-01,
    3.14089276e+00, 3.25899897e-01, 9.65543814e-02,
    3.13528983e+00, 3.24141477e-01, 9.39957941e-02,
    1.55531781e+00,-3.14198542e-03, 1.14360318e-01,
    1.56043061e+00, 1.76807249e-03, 1.10235218e-01,
    -7.07000000e-01, 0.00000000e+00, 0.00000000e+00, 7.07000000e-01,
    1.29888522e-01, 7.69647208e-02, 3.02352183e-02,
    ])
  def cost_func(p):
    loss = []
    DH_params = np.reshape(p[:18], (6,3)) # each link is represented by 3 params
    last_transformation = get_transformation_matrix(p[-7:])
    for m6, cp in zip(list_m6_in_hebi_frame, list_jp):
      ee = calculate_FK_transformation(DH_params, cp)
      ee = ee.dot(last_transformation)
      prediction = ee[0:3, 3].reshape(3)
      loss.append(np.linalg.norm(prediction - m6))
    return np.average(loss)
  return initP, cost_func

def optimize_FK_and_R(initRparam, initFKparam, list_m6, list_jp):
  initP = np.hstack((initRparam, initFKparam)).reshape(-1)
  def cost_func(p):
    loss = []
    R_params = np.reshape(p[:7], -1)
    R = get_transformation_matrix(R_params)
    pad_m6 = np.ones((len(list_m6),4))
    pad_m6[:,0:3] = np.array(list_m6)
    DH_params = np.reshape(p[7:25], (6,3))
    last_transformation = get_transformation_matrix(p[-7:])
    for m6, cp in zip(pad_m6, list_jp):
      ee = calculate_FK_transformation(DH_params, cp)
      ee = ee.dot(last_transformation)
      prediction = ee[0:3, 3].reshape(3)
      loss.append(np.linalg.norm(R.dot(m6)[0:3] - prediction))
    return np.average(loss)
  return initP, cost_func

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

if __name__ == '__main__':
  # Load data from CSV that contains m6 (optitrack tip location) and jp (joint positions)
  df = pd.read_csv('data/old/m6_cp.csv')
  list_m6 = [np.fromstring(r[1:-1], dtype=np.float, sep=' ') for r in df['m6'].to_list()] #[1:-1] to exclude '['']'
  list_jp = [np.fromstring(r[1:-1], dtype=np.float, sep=' ')[0:6]  for r in df['joint_position'].to_list()] # keep only 6 joints
  print("size of datapoints:", len(list_m6))
  print("first m6", list_m6[0])
  print("first jp", list_jp[0])

  # Extract Hebi Default FK EE
  list_hebiee = get_hebi_fk(list_jp, arm_hrdf='/home/hebi/hebi/hebi_ws/src/hebi_teleop/gains/chopstick7D.hrdf')
  # expecting hebiee to be at where the chopstick holder touch the bottom plate, should be defined in arm_container
  list_hebiee_tip = get_hebi_fk_tips(list_hebiee)
  print("first Hebi EE\n", list_hebiee[0])
  print("first Hebi calculated tip\n", list_hebiee_tip[0])

  # dummy params
  R_params, _ = optimize_R_using_hebi_FK(None, None)
  FK_params, _ = optimize_FK_only(None, None)

  # ----------------------------------------------------------------------------
  # STEP1: Optimize R
  # ----------------------------------------------------------------------------
  if False:
    print("\n\nOptimize the transformation matrix R from optitrack frame to hebi\n\n")
    initP, cost_func = optimize_R_using_hebi_FK(list_m6, list_hebiee_tip)
    init_distance = cost_func(initP, verbose=True)
    print('Before optimize, avg distance =', np.average(init_distance))
    print('Before optimize, max distance = ', np.max(init_distance))
    print('Before optimize, the worst datapoint is ', list_m6[np.argmax(init_distance)], list_hebiee_tip[np.argmax(init_distance)])
    # scipy optimize
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=1000, iprint=10)
    est_R = res.x
    est_R[0:4] = np.array(est_R[0:4]) / np.linalg.norm(est_R[0:4]) # normalize quat
    print("Estimated R from optitrack to base", est_R)
    # cmaes optimize
    res = cmaes(cost_func, initP)
    res[0:4] = np.array(res[0:4]) / np.linalg.norm(res)
    print("CMEAS (perhaps more of a global optim)", res)
    R_params = res

  # ----------------------------------------------------------------------------
  # STEP2: Optimize Hebi FK
  # ----------------------------------------------------------------------------
  if False:
    print("\n\nOptimize FK function\n\n")
    R = get_transformation_matrix(R_params)
    list_m6_in_hebi_frame = []
    _m6 = np.ones(4)
    for m6 in list_m6:
      _m6[0:3] = m6
      list_m6_in_hebi_frame.append(R.dot(_m6)[0:3].reshape(3))
    initP, cost_func = optimize_FK_only(list_m6_in_hebi_frame, list_jp)
    print('Before optimize, avg distance =', cost_func(initP))
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=10000, iprint=20).x
    #res = cmaes(cost_func, initP)
    print("Estimated FK cost", cost_func(res))
    print("Estimated FK params", res)
    FK_params = res

  # ----------------------------------------------------------------------------
  # STEP3: Optimize R and FK jointly (or perhaps iteratively?)
  # ----------------------------------------------------------------------------
  if True:
    print("\n\nJointly optimize R and FK locally\n\n")
    initP, cost_func = optimize_FK_and_R(R_params, FK_params, list_m6, list_jp)
    print('Before optimize, avg distance =', cost_func(initP))
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=30000, iprint=50).x
    print('Optimized distance', cost_func(res))
