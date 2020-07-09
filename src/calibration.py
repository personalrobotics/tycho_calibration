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

def get_hebi_fk(joint_positions, arm_hrdf):
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

def get_m6_in_hebi_frame(list_m6, R_params):
  R = get_transformation_matrix(R_params)
  list_m6_in_hebi_frame = []
  _m6 = np.ones(4)
  for m6 in list_m6:
    _m6[0:3] = m6
    list_m6_in_hebi_frame.append(R.dot(_m6)[0:3].reshape(3))
  return list_m6_in_hebi_frame

def get_fk_tips(list_jp, FK_params):
  DH_params = np.reshape(FK_params[:18], (6,3)) # each link is represented by 3 params
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

measured_R = np.array([0, 0, 0, 1, # quat x y z w, almost identity
    -0.025816, -0.479914, -(0.034154-0.009)])

measured_FK = np.array([
     # link twist (alpha); link length (a);  joint offset (d)
     0,       0,        0.101, # 0 2
     np.pi/2, 0,        0.0826, # 3 5
     np.pi,   0.3255,   0.0451, # 6 7
     np.pi,   0.3255,   0.0713,  # 9 10
     np.pi/2, 0,        0.1143, # 12 14
     np.pi/2, 0,        0.1143, # 15 17
     -0.707,  0,   0, 0.707, 0.133, 0.0803, 0.025]) # 22 23 24 # from end to DH to tip

optimized_FK = np.array([
 0.00000000902663874000, 0.00000000000000000000,
 0.09669200429999999513, 1.57079627999999993371,
 0.00000000000000000000, 0.08259997319999999588,
 3.12226253999999991962, 0.32457626000000000532,
 0.04510001979999998800, 3.14159263999999982531,
 0.33232171300000001857, 0.07130004659999999994,
 1.55818177000000002153, 0.00000000000000000000,
 0.11430068999999999646, 1.57079629000000009498,
 0.00000000000000000000, 0.11281335300000000510,
 -0.70699999999999996181, 0.00000000000000000000,
 0.00000000000000000000, 0.70699999999999996181,
 0.13346192200000001060, 0.08030005510000000346,
 0.02351335309999999859])

def optimize_R_using_hebi_FK(list_m6, list_tip, initP=None):
  if initP is None:
    initP = np.array(measured_R)

  def cost_func(p, verbose=False):
    loss = []
    p[0:4] = initP[0:4]
    R = get_transformation_matrix(p)
    _m6 = np.ones(4)
    for m6, hebi_tip in zip(list_m6, list_tip):
      _m6[0:3] = m6
      transform = R.dot(_m6)[0:3]
      loss.append(np.linalg.norm((transform - hebi_tip).reshape(3))) # Euclidean norm
    return np.average(loss) if not verbose else loss
  return initP, cost_func

def optimize_FK_only(list_m6_in_hebi_frame, list_jp, initP=None, sel_params=np.arange(25)):
  if initP is not None:
    defaultP = np.array(initP)
  else:
    defaultP = np.array(optimized_FK)
  initP = defaultP[sel_params]
  def cost_func(_p, verbose=False):
    loss = []
    p = np.array(defaultP)
    p[sel_params] = _p
    p[8] += p[11] - measured_FK[11] + p[5] - measured_FK[5]
    DH_params = p[:18].reshape(6,3)
    last_transformation = get_transformation_matrix(p[-7:])
    for m6, cp in zip(list_m6_in_hebi_frame, list_jp):
      ee = calculate_FK_transformation(DH_params, cp)
      ee = ee.dot(last_transformation)
      prediction = ee[0:3, 3].reshape(3)
      loss.append(np.linalg.norm(prediction - m6))
    deviation_loss = np.sum(np.exp(np.abs(p - measured_FK) * 10) - 1) / len(sel_params) / 10
    print(deviation_loss, np.average(loss))
    return np.average(loss) + deviation_loss if not verbose else loss
  return initP, cost_func

def optimize_FK_and_R(initRparam, initFKparam, list_m6, list_jp):
  initP = np.hstack((initRparam, initFKparam)).reshape(-1)
  def cost_func(p, verbose=True):
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
    return np.average(loss) if not verbose else loss
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

if __name__ == '__main__':
  # Load data from CSV that contains m6 (optitrack tip location) and jp (joint positions)
  df = pd.read_csv('data/m6_jps.csv')
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
  if True:
    print("\n\nOptimize the transformation matrix R from optitrack frame to hebi\n\n")
    initP, cost_func = optimize_R_using_hebi_FK(list_m6, list_hebiee_tip)
    init_distance = cost_func(initP, verbose=True)
    print('Before optimize, avg distance =', np.average(init_distance))
    print('Before optimize, max distance = ', np.max(init_distance))
    print('Before optimize, the worst datapoint is ', list_m6[np.argmax(init_distance)], list_hebiee_tip[np.argmax(init_distance)])
    # scipy optimize
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=1000, iprint=10).x
    est_R = res
    est_R[0:4] = np.array(est_R[0:4]) / np.linalg.norm(est_R[0:4]) # normalize quat
    print("Estimated R from optitrack to base", est_R)
    # cmaes optimize
    #res = cmaes(cost_func, initP)
    #res[0:4] = np.array(res[0:4]) / np.linalg.norm(res)
    #print("CMEAS (perhaps more of a global optim)", res)
    R_params = res

  # ----------------------------------------------------------------------------
  # STEP2: Optimize Hebi FK
  # ----------------------------------------------------------------------------
  if True:
    print("\n\nOptimize FK function\n\n")
    def opt_fk(sel_params):
      print("Optimizing select parameters for FK, sel:", sel_params)
      list_m6_in_hebi_frame = get_m6_in_hebi_frame(list_m6, R_params)
      initP, cost_func = optimize_FK_only(list_m6_in_hebi_frame, list_jp, initP=FK_params, sel_params=sel_params)
      initLoss = cost_func(initP, verbose=True)
      print('Before optimize, avg distance =', np.average(initLoss))
      print('Before optimize, max distance = ', np.max(initLoss))
      res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=2000, iprint=20).x
      newCost = cost_func(res, verbose=True)
      print('After optimize, avg distance =', np.average(newCost))
      print('After optimize, max distance = ', np.max(newCost))
      return res

    #b = [1,4,8,11,13,16,18,19,20,21] # not optimizing
    a = [0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 22, 23, 24]
    for opt_params in [a]:
      new_FK_params = opt_fk(opt_params)
      FK_params[opt_params] = new_FK_params
      FK_params[8] += FK_params[5] - measured_FK[5] + FK_params[11] - measured_FK[11]

    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.20f},'.format}, linewidth=50)
    print(FK_params)
    print("Changes")
    print(FK_params - measured_FK)

  # ----------------------------------------------------------------------------
  # STEP3: Optimize R and FK iteratively
  # ----------------------------------------------------------------------------
  if False:
      print("\n\nOptimize R and FK iteratively")
      initP, cost_func = optimize_FK_and_R(R_params, FK_params, list_m6, list_jp)
      initCost = cost_func(initP, verbose=True)
      print("Before optimization, avg distance", np.average(initCost))
      print("Max distance", np.max(initCost))
      for _ in range(1):
        list_fk_tips = get_fk_tips(list_jp, FK_params)
        _, cost_func = optimize_R_using_hebi_FK(list_m6, list_fk_tips)
        R_params = scipy_optimize(cost_func, R_params, method='L-BFGS-B', max_func=1000, iprint=50).x
        R_params[0:4] = R_params[:4] / np.linalg.norm(R_params[:4])
        print('New R params', R_params)
        newCost = cost_func(R_params, verbose=True)
        print("New average distance", np.average(newCost))
        print("Max distance", np.max(newCost))
        list_m6_in_hebi_frame = get_m6_in_hebi_frame(list_m6, R_params)
        _, cost_func = optimize_FK_only(list_m6_in_hebi_frame, list_jp)
        FK_params = scipy_optimize(cost_func, FK_params, method='L-BFGS-B', max_func=1000, iprint=50).x
        newP, cost_func = optimize_FK_and_R(R_params, FK_params, list_m6, list_jp)
        newCost = cost_func(newP, verbose=True)
        print("New average distance", np.average(newCost))
        print("Max distance", np.max(newCost))

  # ----------------------------------------------------------------------------
  # ??? STEP4: Optimize R and FK jointly
  # ----------------------------------------------------------------------------
  if False:
    print("\n\nJointly optimize R and FK\n\n")
    initP, cost_func = optimize_FK_and_R(R_params, FK_params, list_m6, list_jp)
    print('Before optimize, avg distance =', cost_func(initP))
    res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=30000, iprint=50).x
    print('Optimized distance', cost_func(res))
