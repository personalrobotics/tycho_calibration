import numpy as np
import scipy
from scipy.spatial.transform import Rotation as scipyR

# We measure the position of four optitrack balls under hebi frame
hebi_locs = [
  [0, -0.1, 0.004], #
  [-0.0355, -0.473, -0.01],
  [-0.0355, -0.273, -0.01],
  [-0.1855, -0.473, -0.01]
]

optitrack_locs = [
  [0.025726, 0.378255, 0.030150, 1],
  [-0.009739, 0.009678, 0.017897, 1],
  [-0.010010, 0.209811, 0.019007, 1],
  [-0.15956, 0.009344, 0.019001, 1]
]

def get_transformation_matrix(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_dcm() # scipy >=1.4.0 will always normalize quat
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def cost_func(p, verbose=False):
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
     -0.025816, -0.479914, -(0.034154-0.009)]) # measured x y z

print('InitP performance:', cost_func(initP, verbose=True))
res = cmaes(cost_func, initP)
res[0:4] = res[0:4] / np.linalg.norm(res[0:4])
print(res)
print(cost_func(res, verbose=True))

#[ 9.92209436e-04 -1.32798491e-03 -3.34384441e-03  9.99993035e-01
# -1.47798874e+00 -5.50633367e-01 -3.26888549e-02]
