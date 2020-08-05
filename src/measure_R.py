# We place a few mocap balls around hebi base, record the optitrack locations
# and use ruler to measure the ball location under hebi's frame. This script
# accepts a transformation from optitrack to hebi's frame and evaluates it.

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as scipyR

# We measure the position of four optitrack balls under hebi frame

hebi_locs =np.array( [
  [-0.18649, -0.06, 0.00922], #
  [0.01351, -0.06, 0.00922],
  [0.01351, -0.21, 0.00922],
  [0.01501, -0.30,0.03122],
  [0.01501, -0.35,0.03122],
  [-0.061,-0.006,0.0811],
])

# read from optitrack
optitrack_locs =np.array([
[0.010005, 0.000001, 0.210172],
[0.010000,0.000001,0.010013],
[0.159879,0.000001,0.010147],
[0.248919,0.023824,0.008006],
[0.297900,0.023164,0.006664],
[-0.044225,0.069767,0.084373],
])
new_order=[2,0,1]

optitrack_locs=optitrack_locs[:,new_order]
optitrack_locs[:,0]=-optitrack_locs[:,0]
optitrack_locs[:,1]=-optitrack_locs[:,1]
print(optitrack_locs)
optitrack_locs = np.hstack((optitrack_locs, np.ones(optitrack_locs.shape[0]).reshape(-1,1)))

def get_transformation_matrix(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  qx,qy,qz,qw = 0,0,0,1
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_dcm() # scipy >=1.4.0 will always normalize quat
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def cost_func(p, verbose=False):
  p[0:4] = 0,0,0,1
  R = get_transformation_matrix(p)
  loss = []
  for op, hp in zip(optitrack_locs, hebi_locs):
    loss.append(np.linalg.norm(R.dot(op)[0:3] - hp))
    if verbose:
      print(R.dot(op)[0:3] , hp)
  return loss if verbose else np.average(loss)

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
      #print('CMAES found a new set of best params, achieving', best_so_far)
      #print('params', best_params)
    es.logger.add()
    es.disp()
  es.result_pretty()
  return best_params

initP = np.array([0, 0, 0, 1, # quat x y z w, almost identity
                  0.024, -0.05, 0.01059545]) # measured by hand

print('InitP performance:', cost_func(initP, verbose=True))

#res = cmaes(cost_func, initP)
#res[0:4] = res[0:4] / np.linalg.norm(res[0:4])
#print(res)
#print(cost_func(res, verbose=True))