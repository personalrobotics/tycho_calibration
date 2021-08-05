#===============================================================================
# Visualize the collision balls on top of an interactive URDF model
# ------------------------------------------------------------------------------
# For fast collision checking, we create spheres covering the robot body.
# At runtime, we compute the positions of spheres using our own FK parameters
# and an "extra link" of transformation for each sphere.
# This script allows you to load and visualize an URDF model, move the robot,
# refresh the positions of spheres and adjust the individual link for each ball.
#
# To use:
#   0.  Install pybullet.
#   1.  Set URDF_PATH = '...' to the full path. If you used an URDF compiled by
#       ROS, you might have to remove 'packages://tycho_description' from urdf.
#   2.  Load all the DH parameters.
#   3.  Launch the script, you should see the robot and spheres (if any).
#   4.  Use mouse to move the robot around, press 'r' to refresh spheres.
#   5.  In the GUI, press
#       'n' to select spheres;
#       'o' to change color of the current selection;
#       '123456' to move;
#       '78' to change ball size;
#       '90' to adjust refinement;
#       'a' to print the current sphere config (the script will not auto save);
#       'q' to quit.
#
#===============================================================================

URDF_PATH = '/home/prl/tycho_ws/src/tycho_env/tycho_env/assets/hebi_pybullet.urdf'
JOINT_IDX = [2, 4, 6, 8, 10, 12, 14]
SPHERE_COLOR = [144.0/255,238.0/255,144.0/255, 0.5]
HIGHLIGHT_COLOR = [1,1,1,0.5]

from tycho_env.utils import DH_params, MOVING_POSITION, get_DH_transformation, get_transformation_matrix, calculate_FK_transformation
import pybullet as p
import time
import numpy as np

class Sphere:
    def __init__(self, center, radius):
        self._center = np.asarray(center)
        self._radius = radius

    @property
    def center(self):
        return self._center.tolist()

    @property
    def radius(self):
        return self._radius


class Bullet:

    def __init__(self, gui=False):
        self.use_gui = gui
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.urdf_path = None
        self.obstacle_ids = []
        self.obstacle_collision_ids = []
        self.sphere_configs = []

    def __del__(self):
        p.disconnect(self.clid)

    def reload(self):
        p.disconnect(self.clid)
        self.clid = p.connect(p.GUI)
        if self.urdf_path is not None:
            self.load_robot(self.urdf_path)
        self.obstacle_ids = []

    def load_robot(self, path):
        self.robot_id = p.loadURDF(path, useFixedBase=True, physicsClientId=self.clid)
        self.urdf_path = path

    def load_spheres(self, sphere_configs):
        self.sphere_configs = sphere_configs

    def show_spheres(self, jp):

        spheres = []
        for s in self.sphere_configs:
            loc = get_sphere_loc(jp, s[:-1])
            spheres.append(Sphere(loc, s[-1]))

        ids = []
        for sphere in spheres:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                rgbaColor=SPHERE_COLOR,
                physicsClientId=self.clid,
            )
            #obstacle_collision_id = p.createCollisionShape(
            #    shapeType=p.GEOM_SPHERE,
            #    radius=sphere.radius,
            #    physicsClientId=self.clid,
            #)
            obstacle_id = p.createMultiBody(
                basePosition=sphere.center,
                baseVisualShapeIndex=obstacle_visual_id,
                #baseCollisionShapeIndex=obstacle_collision_id,
                physicsClientId=self.clid,
            )
            ids.append(obstacle_id)

        self.obstacle_ids.extend(ids)
        return ids

    def clear_all_obstacles(self):
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []

    def marionette(self, config):

        for i, idx in enumerate(JOINT_IDX):
            p.resetJointState(
                self.robot_id, idx, config[i], physicsClientId=self.clid
            )

    def get_joint_positions(self):
        joint_positions = np.zeros(7)
        for i in range(7):
            joint_positions[i] = p.getJointState(self.robot_id, JOINT_IDX[i])[0]
        return joint_positions



def get_sphere_loc(joint_position, sphere_config):
    joint_id = sphere_config[0]
    last_transformation = get_transformation_matrix(sphere_config[1])

    ee = np.eye(4)
    for (alpha, a, d), theta in zip(DH_params[:joint_id,:], joint_position[:joint_id]):
      ee = ee.dot(get_DH_transformation(alpha, a, theta, d))
    ee = ee.dot(last_transformation)
    return ee[0:3, 3].reshape(3)

# Use n to select the next sphere
if __name__ == '__main__':
    my_bullet = Bullet(gui=True)
    my_bullet.load_robot(URDF_PATH)

    joint_positions = [MOVING_POSITION]

    sphere_configs = [
    # Link, transform, radius
     [0, [1,0,0,0,-0.02,0,0.02], 0.065],
     [2, [1,0,0,0,0.02,0,0], 0.105],
     [2, [1,0,0,0,0.15,0,0.02], 0.04],
     [2, [1,0,0,0,0.22,0,0.02], 0.04],
     [2, [1,0,0,0,0.305,0,-0.02], 0.075],
     [3, [1,0,0,0,0.07,0,0.02], 0.035],
     [3, [1,0,0,0,0.131,0,0.02], 0.035],
     [3, [1,0,0,0,0.1845,0,0.02], 0.035],
     [3, [1,0,0,0,0.2415,0,0.02], 0.035],
     [3, [1,0,0,0,0.30575,0,-0.01125], 0.061],
     [4, [1,0,0,0,-0.005,-0.055,-0.01], 0.0555],
     [4, [1,0,0,0,-0.04,-0.055,-0.01], 0.0455],
     [5, [1,0,0,0, 0.01,-0.05,0], 0.05],
     [5, [1,0,0,0, -0.03875,-0.055,-0.00125], 0.0425],
     [6, [1,0,0,0, 0.0, 0.04375,-0.0025], 0.05],
     [6, [1,0,0,0, -0.045, 0.06, 0.0025], 0.04],
     [6, [1,0,0,0, 0.118368, 0.07914739999999999, 0.024354999999999995], 0.0025],
     [6, [1,0,0,0, -0.1181944999999989, 0.07914739999999999, 0.024354999999999995], 0.003],
     [7, [1, 0, 0, 0, 0.11593750000000001, 0.0525, 0.023125], 0.0025],
     [7, [1, 0, 0, 0, -0.09569449999999888, -0.04374999999999999, 0.0225], 0.00375],
    ]

    tid = None

    refinement = 0.01

    my_bullet.load_spheres(sphere_configs)

    for jp in joint_positions:
        my_bullet.marionette(jp)

        cursor = 0

        my_bullet.show_spheres(jp)

        keys = ''
        while True:
            keys = p.getKeyboardEvents()
            p.stepSimulation()
            time.sleep(0.01)

            test_keys = ['q','n','1','2','3','4','5','6','7','8','o','a','r','9','0']
            pressed_key = None
            for _k in test_keys:
                if ord(_k) in keys:
                    state = keys[ord(_k)]
                    if (state & p.KEY_WAS_RELEASED):
                        pressed_key = _k
                        break

            if pressed_key == None:
                continue
            elif pressed_key == 'q':
                break
            elif pressed_key == 'n':
                cursor += 1
            elif pressed_key == 'r':
                jp = my_bullet.get_joint_positions()
                my_bullet.clear_all_obstacles()
                my_bullet.show_spheres(jp)
            elif pressed_key in ['9','0']:
                if pressed_key == '9':
                    refinement = refinement * 2
                else:
                    refinement = refinement / 2
                print("New refinement", refinement)
            else:

                cursor = cursor % len(my_bullet.obstacle_ids)
                obj_id = my_bullet.obstacle_ids[cursor]
                sconfig = my_bullet.sphere_configs[cursor]

                if pressed_key == 'a':
                    print(obj_id, sconfig)


                elif pressed_key in ['1','2','3','4','5','6']:
                    shift_idx = (int(pressed_key) -1 ) // 2
                    shift_amount = refinement * (int(pressed_key) % 2 - 0.5)
                    sconfig[1][4 + shift_idx] += shift_amount
                    new_loc = get_sphere_loc(jp, sconfig[:-1])
                    p.resetBasePositionAndOrientation(obj_id, new_loc,[0,0,0,1])

                elif pressed_key in ['7', '8']:
                    sconfig[-1] += refinement * (int(pressed_key) - 7.5)
                    my_bullet.clear_all_obstacles()
                    my_bullet.show_spheres(jp)

                elif pressed_key == 'o':
                    c = p.getVisualShapeData(obj_id)[0][7]
                    p.changeVisualShape(obj_id, -1,
                        rgbaColor=(HIGHLIGHT_COLOR if c[0] < 1.
                                                   else SPHERE_COLOR))


    my_bullet.clear_all_obstacles()
