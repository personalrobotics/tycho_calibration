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

def calculate_FK_transformation(FKparams, current_position):
  # Given a list of FKparams, shape N by 3, return transformation
  ee = np.eye(4)
  for (alpha, a, d), theta in zip(FKparams, current_position):
    ee = ee.dot(get_DH_transformation(alpha, a, theta, d))
  return ee

def optimize_R_using_hebi_FK(list_m6, list_hebiee_tip, initP=None):
  initP = initP or np.array(
    [0.001, 0.001, 0.001, 1.000, # quat x y z w, almost identity
     -1.4426,-0.5949,0.0058] # x y z shift
  )
  def cost_func(p):
    loss = 0
    R = get_transformation_matrix(p)
    _m6 = np.ones(4)
    for m6, hebi_tip in zip(list_m6, list_hebiee_tip):
      _m6[0:3] = m6
      transform = R.dot(_m6)[0:3]
      loss += np.linalg.norm((transform - hebi_tip).reshape(3))
    return loss / len(list_m6)
  return initP, cost_func

def optimize_FK_only(initP=None):
  initP = initP or np.array([
    np.pi/2,0.01,0.01,
    np.pi/2,0.01,0.01,
    np.pi/2,0.01,0.01,
    np.pi/2,0.01,0.01,
    np.pi/2,0.01,0.01,
    np.pi/2,0.01,0.01,
    np.pi/2,0.01,0.01])
  def cost_func(data, p):
    loss = []
    params = np.reshape(p, (-1,3)) # each link is represented by 3 params
    for m6, cp in data:
      ee = calculate_FK_transformation(params, cp)
      prediction = ee[0:3, 3].reshape(3)
      loss.append(np.linalg.norm(prediction - m6))
    return np.average(loss)
  return initP, cost_func

def optimize_joint(initRparam, initFKparam):
  pass

def print_error(error):
  square_error = np.square(error)
  print('Average of squared error per each dimension', np.average(square_error, axis=0))
  print('Average distance error', np.average(np.linalg.norm(error, axis=1)))

def get_hebi_fk(joint_positions,
                arm_hrdf):
  from hebi_env.arm_container import create_empty_robot
  arm = create_empty_robot(arm_hrdf)
  return np.array([arm.get_FK_ee(p) for p in joint_positions]) # data_size x 4 x 4

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

def cmaes(func, initP):
  import cma
  es = cma.CMAEvolutionStrategy(initP, 5)
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
  df = pd.read_csv('data/stored_m6&cp.csv')
  list_m6 = [np.fromstring(r[1:-1], dtype=np.float, sep=',') for r in df['m6'].to_list()] #[1:-1] to exclude '['']'
  list_jp = [np.fromstring(r[1:-1], dtype=np.float, sep=',')  for r in df['current_position'].to_list()]
  print("size of datapoints:", len(list_m6))
  print("first m6", list_m6[0])
  print("first jp", list_jp[0])

  # Extract Hebi Default FK EE
  list_hebiee = get_hebi_fk(list_jp, arm_hrdf='/home/hebi/hebi/hebi_ws/src/hebi_teleop/gains/chopstick7D.hrdf')
  # expecting hebiee to be at where the chopstick holder touch the bottom plate, should be defined in arm_container
  list_hebiee_tip = get_hebi_fk_tips(list_hebiee)
  print("first Hebi EE\n", list_hebiee[0])
  print("first Hebi calculated tip\n", list_hebiee_tip[0])

  # Optimize R
  print("\n\nOptimize the transformation matrix R from optitrack frame to hebi\n\n")
  initP, cost_func = optimize_R_using_hebi_FK(list_m6, list_hebiee_tip)
  print('Before optimize, func =', cost_func(initP))
  res = scipy_optimize(cost_func, initP, method='L-BFGS-B', max_func=1000, iprint=10)
  #res = cmaes(cost_func, initP)

  # Optimize hebi's FK
  print("\n\nOptimize FK function\n\n")
  #initP, cost_func = optimize_FK()
  #res = BLABLA

  # Last joint optimize
  print("\n\nJointly optimize R and FK locally\n\n")
  #initP, cost_func = optimize_FKandR()
  # res = BLABLA

  # Profile Hebi's performance
  #print("Profiling hebi's performance")
  #print(profile_hebi(data))

  # Initial params
  #func = partial(cost_func, data)
  #print('Inital matrix loss',func(initP))


  #np.savetxt('optimized_FK.txt',optimize_FK_param, delimiter=',')

