# Assumine you have a recording of a tracked ball, where the ball rotates around the center of robot and produce an arc of a circle.
# We will try to find the center and the rotation of the base frame accordingly.
# Notice that the z (height) cannot be estimated solely from this script.

from __future__ import print_function

import os
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial.transform import Rotation as scipyR
from functools import partial

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def start_colorful():
    print(bcolors.OKGREEN)
def stop_colorful():
    print(bcolors.ENDC)

def print_info(x, prefix="", file=None):
  print("{} Average {}, STD {}. Min {}. Max {}".format(
    prefix, np.average(x), np.std(x), np.min(x), np.max(x)), file=file)

# ==============================================================
# Utilities for cost functoolsnction
# ==============================================================

def dis_point_to_plane(point, plane):
  # Refere to https://mathinsight.org/distance_point_plane
  # Deprecated
  x, y, z = point
  a,b,c,d = plane
  return np.abs(a*x + b*y + c*z + d) / np.sqrt( a*a + b*b + c*c + 1e-10)

def height_cost_fn(list_ball, params, verbose=False):
  # A cost function that considers the avg distance between points to the plane
  # Deprecated
  costs = np.array([dis_point_to_plane(ball, params) for ball in list_ball])
  idx = not_outlier(costs)
  return np.average(costs[idx]) if not verbose else costs

def get_transformation_matrix(plane_params, shift_params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return the transformation matrix
  qx,qy,qz,qw = plane_params
  x,y,z = shift_params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_matrix()
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def not_outlier(arr, allowable_deviation=2):
  mean = np.mean(arr)
  std = np.std(arr)
  dis = abs(arr - mean)
  not_outlier =  dis < allowable_deviation * std
  return not_outlier

def generate_cost_func(list_ball, default_plane_params, sel_param_idx,
                       coefficient_of_projected_radius_std=1,
                       coefficient_of_projected_heights_std=10,
                       coefficient_of_deviation=0.05):
  # A cost function that considers the both the avg distance and the consistence of radius
  points = np.ones((len(list_ball), 4))
  points[:, 0:3] = list_ball

  initP = default_plane_params[sel_param_idx]

  def cost_func(new_params, verbose=False):
    plane_params = np.array(default_plane_params)
    plane_params[sel_param_idx] = new_params
    R = get_transformation_matrix(plane_params[0:4], plane_params[4:7])
    transformed_points = np.array([R.dot(p)[0:3] for p in points])
    heights = np.array([p[2] for p in transformed_points])
    idx = not_outlier(heights)
    heights = heights[idx]
    projected_points = [(p[0],p[1]) for p in transformed_points[idx]]
    projected_distance = [np.linalg.norm(p) for p in projected_points]
    deviation = np.linalg.norm(plane_params[0:4] / np.linalg.norm(plane_params[0:4]) - np.array([0,0,0,1]))
    # The deviation cost assumes that we should be close to rotation default (0,0,0,1).
    # Without it, CMAES can go wild and flip the rotation by 180.
    cost = deviation * coefficient_of_deviation + \
           coefficient_of_projected_radius_std * np.std(projected_distance) + \
           coefficient_of_projected_heights_std * np.std(heights)
    return (projected_distance, heights, deviation) if verbose is True else cost

  return initP, cost_func

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

import argparse
def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default="data/calibration-fitR.csv")
    parser.add_argument('--cmaes', action='store_true')
    parser.add_argument('--plt', action='store_true')
    return parser

if __name__ == '__main__':
  args = construct_parser().parse_args()

  from datetime import datetime
  foldername = str(datetime.now().strftime("fitR-%m%d-%H-%M")) + ("CMAES" if args.cmaes else "LBFGS")
  os.mkdir(foldername)
  f = open(os.path.join(foldername,'eval.txt'),'a')
  print("Datafile: ", args.csv, "\nCMAES? ", args.cmaes, file=f)

  np.set_printoptions(suppress=True, formatter={'float_kind':'{:.8f},'.format}, linewidth=80)

  # Load data from CSV that contains the ball location and the joint positions
  df = pd.read_csv(args.csv)
  optim_func = cmaes if args.cmaes else scipy_optimize
  list_ball = [np.fromstring(r[1:-1], dtype=np.float, sep=' ') for r in df['ball_loc'].to_list()] #[1:-1] to exclude '['']'
  ###########################################
  ### TAKE ONLY THE DATA YOU NEEDED
  ###########################################
  list_ball = list_ball[100:2200] + list_ball[2550:-100]
  print(bcolors.OKGREEN, "\nNumber of data points to use:", len(list_ball), bcolors.ENDC)
  print("Number of data points to use:", len(list_ball), file=f)

  # Approach 1: Closed form solution .... minimize *vertical* distance
  # A = np.array(list_ball)
  # A[:, 2] = 1
  # B = np.array([m[2] for m in list_ball]).reshape(-1)
  # res = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), B)
  # print(res)
  # a,b,c = res
  # costs = [a*x + b*y + c - z for (x,y,z) in list_ball]
  # print(np.average(costs), np.max(costs))

  # Approach 2: First find the plane of the circle. Then optimize the center of the circle. Then joint optimize.
  heights = np.array([ball_loc[2] for ball_loc in list_ball]).reshape(-1)
  avg_heights = np.average(heights)

  cost_func = partial(height_cost_fn, list_ball)
  initP = np.array([0, 0, 0, np.average(heights)])
  plane_params = optim_func(cost_func, initP)
  newCost = cost_func(plane_params, verbose=True)
  start_colorful()
  print('For reference, this is a plane that will minimize the *perpendicular* distance from points to the plane')
  print('The plane is Ax + By + Cz + D = 0:', plane_params)
  print_info(newCost, "costs")
  print("Found a plane (ax+by+cz+d=0)", plane_params, file=f)
  print_info(newCost, "costs", file=f)
  stop_colorful()

  init_plane_params = np.zeros(7)
  init_plane_params[3] = 1
  initP, cost_func = generate_cost_func(list_ball, init_plane_params, [0,1,2,3],
    coefficient_of_projected_radius_std=0, coefficient_of_projected_heights_std=10)
  rotate_params = optim_func(cost_func, initP)
  newCost = cost_func(rotate_params, verbose=True)[1]
  start_colorful()
  print('This is a plane that will minimize the *std* of the heights of points after transformation')
  print('The plane is Qx Qy Qz Qw:', rotate_params)
  print_info(newCost, "costs")
  print("Found a plane (qx qy qz qw)", rotate_params, file=f)
  print_info(newCost, "costs", file=f)
  stop_colorful()

  #sinit_plane_params[0:4] = rotate_params
  init_plane_params[0:4] = [0, 0, 0, 1]
  init_plane_params[4:7] = [-1.07, 0.08, avg_heights]
  initP, cost_func = generate_cost_func(list_ball, init_plane_params, [4,5,6],
    coefficient_of_projected_radius_std=1, coefficient_of_projected_heights_std=0)
  shift_params = optim_func(cost_func, initP)
  new_radius = cost_func(shift_params, verbose=True)[0]
  start_colorful()
  print('Found the center of circle to minimize the std of radius given the rotation')
  print(shift_params)
  print_info(new_radius, "radius")
  print("Found the center of circle", shift_params, file=f)
  print_info(new_radius, "radius", file=f)
  stop_colorful()

  init_plane_params[4:7] = shift_params
  initP, cost_func = generate_cost_func(list_ball, init_plane_params, np.arange(7),
    coefficient_of_projected_radius_std=1, coefficient_of_projected_heights_std=10)
  res = optim_func(cost_func, initP)
  res[0:4] = res[0:4] / np.linalg.norm(res[0:4])
  start_colorful()
  print('Optimized Rotation', res)
  print('Optimized Rotation', res, file=f)
  print('Rotation in matrix form', scipyR.from_quat(res[0:4]).as_matrix())
  print('Rotation in quaternion', res[0:4])

  proj_r, heights, deviation = cost_func(res, verbose=True)
  print_info(proj_r, "proj_r")
  print_info(heights, "proj_heights")
  print("deviation", deviation)
  print_info(proj_r, "proj_r", file=f)
  print_info(heights, "proj_heights", file=f)
  print("deviation", deviation, file=f)
  stop_colorful()

  f.close()

  import seaborn as sns
  import matplotlib.pyplot as plt
  plt.figure(0)
  sns.distplot(heights)
  plt.title("Projected heights")
  plt.xlabel("Heights")
  plt.ylabel("# of occurence")
  plt.savefig(os.path.join(foldername, "fit_R_heights_dist.jpg"))
  plt.show() if not args.plt else print("Saved plot fit_R_heights_dist")

  plt.figure(1)
  sns.distplot(proj_r)
  plt.title("Projected distance to center")
  plt.xlabel("Distance to center")
  plt.ylabel("# of occurence")
  plt.savefig(os.path.join(foldername, "fit_R_radius_dist.jpg"))
  plt.show() if not args.plt else print("Saved plot fit_R_radius_dist")

  plt.figure(2)
  x = np.arange(len(heights))
  sns.jointplot(x=x, y=heights)
  plt.title("Height in order of recording")
  plt.xlabel("Recorded point #")
  plt.ylabel("heights")
  plt.savefig(os.path.join(foldername, "fit_R_heights.jpg"))
  plt.show() if not args.plt else print("Saved plot fit_R_heights")

  plt.figure(3)
  sns.jointplot(x=x, y=proj_r)
  plt.title("Projected distance to center, in order of recording")
  plt.xlabel("Recorded point #")
  plt.ylabel("dis")
  plt.savefig(os.path.join(foldername, "fit_R_radius.jpg"))
  plt.show() if not args.plt else print("Saved plot fit_R_radius")
