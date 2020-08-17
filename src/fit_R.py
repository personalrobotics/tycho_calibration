# Assumine you have a recording of M6, where M6 rotates around the center of robot and produce an arc of a circle.
# We will try to find the center and the rotation fo the base frame accordingly.
# Notice that the z (height) cannot be estimated solely from this script.

from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.transform import Rotation as scipyR
from functools import partial

# ==============================================================
# Utilities for cost functoolsnction
# ==============================================================

def dis_point_to_plane(point, plane):
  x, y, z = point
  a,b,c,d = plane
  return np.abs(a*x + b*y + c*z + d) / np.sqrt( a*a + b*b + c*c + 1e-10)

def get_transformation_matrix(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_dcm() # scipy >=1.4.0 will always normalize quat
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def cost_fn(list_m6, params, verbose=False):
  costs = [dis_point_to_plane(m6, params) for m6 in list_m6]
  return np.average(costs) if not verbose else costs

def fancy_cost_fn(list_m6, params, verbose=None):
  points = np.ones((len(list_m6), 4))
  points[:, 0:3] = list_m6
  R = get_transformation_matrix(params)
  transformed_points = [R.dot(p)[0:3] for p in points]
  heights = [p[2] for p in transformed_points]
  projected_points = [(p[0],p[1]) for p in transformed_points]
  projected_distance = [np.linalg.norm(p) for p in projected_points]
  #print(np.std(projected_distance), np.max(np.abs(projected_distance - np.average(projected_distance))),
  #      np.std(heights), np.max(np.abs(heights - np.average(heights))))
  fancy_cost = np.std(projected_distance) + 10 * np.std(heights)
  return (projected_distance, heights) if verbose is True else fancy_cost

def scipy_optimize(func, initP, method='L-BFGS-B', max_func=15000, iprint=1, save=None):
  res = scipy.optimize.minimize(func, initP, method=method, options={'disp': None, 'maxfun': max_func, 'iprint': iprint})
  print('After optimize, minimum=', func(res.x))
  print("Scipy optimized params", res.x)
  (save and np.savetxt('results/'+save, res.x, delimiter=',',fmt='%f'))
  return res.x

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

if __name__ == '__main__':
  np.set_printoptions(suppress=True, formatter={'float_kind':'{:.8f},'.format}, linewidth=80)

  # Load data from CSV that contains m6 (optitrack tip location) and jp (joint positions)
  df = pd.read_csv('data/m6_circle.csv')
  list_m6 = [np.fromstring(r[1:-1], dtype=np.float, sep=' ') for r in df['m6'].to_list()] #[1:-1] to exclude '['']'
  list_m6 = list_m6[500:-500]
  cost_func = partial(cost_fn, list_m6)

  # Approach 1: Closed form solution .... minimize *vertical* distance
  # A = np.array(list_m6)
  # A[:, 2] = 1
  # B = np.array([m[2] for m in list_m6]).reshape(-1)
  # res = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), B)
  # print(res)
  # a,b,c = res
  # costs = [a*x + b*y + c - z for (x,y,z) in list_m6]
  # print(np.average(costs), np.max(costs))

  # Approach 2: Optimization solution .. minimize *perpendicular* distance
  # heights = np.array([m[2] for m in list_m6]).reshape(-1)
  # initP = np.array([0, 0, 0, np.average(heights)])
  # res = scipy_optimize(cost_func, initP)
  # newCost = cost_func(res, verbose=True)
  # print('Avg cost {} Max cost {}'.format(np.average(newCost),np.max(newCost)))

  # Approach 3: Minimize projected distance and heights
  heights = np.array([m[2] for m in list_m6]).reshape(-1)
  initP = np.array([0,0,0,1, -1.07 , 0.08, np.average(heights)])
  fancy_cost_func = partial(fancy_cost_fn, list_m6)
  res = scipy_optimize(fancy_cost_func, initP)
  #res = cmaes(fancy_cost_func, initP)
  res[0:4] = res[0:4] / np.linalg.norm(res[0:4])
  print('Optimized R', res)
  print('rotation matrix')
  print(scipyR.from_quat(res[0:4]).as_dcm())
  print('in quaternion:',res[0:4])

  proj_dis, heights = fancy_cost_func(res, verbose=True)
  print('proj_dis avg {} max {}'.format(np.average(proj_dis),np.max(proj_dis)))
  print('heights avg {} max {}'.format(np.average(heights),np.max(heights)))

  import seaborn as sns
  import matplotlib.pyplot as plt
  sns.distplot(heights)
  plt.title("Projected heights")
  plt.show()

  x = np.arange(len(heights))
  sns.jointplot(x=x, y=heights)
  plt.title("Height per point")
  plt.show()

  sns.distplot(proj_dis)
  plt.title("Projected distance")
  plt.show()

  sns.jointplot(x=x, y=proj_dis)
  plt.title("Projected distance to center of circle, per point")
  plt.show()
