import pybullet as p
import time
from viz_collision_balls import Bullet, URDF_PATH
from tycho_env import TychoEnv
from tycho_env.utils import MOVING_POSITION
import pandas as pd
import numpy as np

my_bullet = Bullet(gui=True)
my_bullet.load_robot(URDF_PATH)
my_bullet.marionette(MOVING_POSITION)

keys = ''
path = "/home/prl/tycho_ws/src/tycho_calibration/output.csv"

df = pd.read_csv(path)
list_ball = [np.fromstring(r[1:-1], dtype=np.float, sep=' ') for r in df['ball_loc'].to_list()] #[1:-1] to exclude '['']'
list_jp = [np.fromstring(r[1:-1], dtype=np.float, sep=' ')  for r in df['joint_position'].to_list()] # keep only 6 joints

for i, jp in enumerate(list_jp):
    print(i)
    # keys = p.getKeyboardEvents()
    p.stepSimulation()
    time.sleep(0.01)
    my_bullet.marionette(jp)

