import glob, os, sys, csv, cv2, time, numpy as np
from scipy.spatial.transform import Rotation as scipyR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_folder_location', type=str, default="/home/prl/tycho_ws/src/tycho_teleop/recording/")
    parser.add_argument('--bag_folder_name', type=str, default="calibration")
    parser.add_argument('--save_file_name', type=str, default="data/ball_and_jointpos.csv")
    parser.add_argument('-r','--cut_off', action='store_true')
    return parser

CUTOFF_BEGINNING = 3 * 10^8                    # unit in nanoseconds (10^9 nanoseconds = 1 second)
CUTOFF_LENGTH = CUTOFF_BEGINNING + 5 * 10^8   # unit in nanoseconds


args = construct_parser().parse_args()
assert(args.bag_folder_name != '')
assert(args.save_file_name != '')
args.bag_folder = os.path.join(args.bag_folder_location, args.bag_folder_name)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
print(bcolors.BOLD + bcolors.FAIL + "[WARNING]: PLEASE CLOSE ANY ROS TOPIC !!!!!" + bcolors.ENDC)

from subprocess import Popen, STDOUT, PIPE
sys.path.append("/usr/lib/python3/dist-packages")
import rospy
import message_filters
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import JointState

FNULL = open(os.devnull, 'w')

label_name = args.save_file_name
print(label_name)

if os.path.exists(label_name):
    print("Label file or image folder already exists!"
          "Quit now to prevent overwriting."
          "Press Enter to continue...")
    input()
label_file = open(label_name, mode='w')
writer = csv.writer(label_file, delimiter = ',')


datapoint_count = 0
trajectory_count = 0
rosbag_name = ""

def write_csv_header(writer):
    #misc_fieldnames = [[['{}_{}{}'.format(_t,_pve,i) for i in range(7)] for _pve in ['p','v','e']] for _t in ['state','cmd']]
    #misc_fieldnames = [sum(x, []) for x in misc_fieldnames]
    #misc_fieldnames = sum(misc_fieldnames, [])
    fieldnames = ['id', 'joint_position', 'ball_loc'] #+ misc_fieldnames
    writer.writerow(fieldnames)

def start_rosbag_play(file):
  print('Playing bags with prefix {}'.format(os.path.basename(file)))
  args1 = ['rosbag', 'play', file]
  # if any of the bag file is missing the command will quit immediately
  return Popen(args1, stdout=FNULL, stderr=STDOUT)

def callback(*argv):
    global datapoint_count
    global rosbag_name
    global start_timer

    joint_state, ball_msg = argv[:2]
    my_timestamp = ball_msg.header.stamp

    if args.cut_off:
        if start_timer is None:
            start_timer = my_timestamp
        elapsed_time = (my_timestamp - start_timer).to_sec()
        if elapsed_time < CUTOFF_BEGINNING or elapsed_time > CUTOFF_LENGTH:
            return

    ball_point = ball_msg.point
    ball_loc = np.array([ball_point.x, ball_point.y, ball_point.z])
    joint_position=np.array([m for m in joint_state.position])
    rowinfo = ([datapoint_count]
                + [np.array2string(joint_position, precision=7, separator=' ', max_line_width=9999)] + [np.array2string(ball_loc, precision=6, separator=' ', max_line_width=9999)])
                #+ list(state_msg.position) + list(state_msg.velocity) + list(state_msg.effort)
                #+ list(command_msg.position) + list(command_msg.velocity) + list(command_msg.effort))

    writer.writerow(rowinfo)
    datapoint_count += 1

def init_subscriber():
    rospy.init_node('syncCalibrationData')
    choppose_sub = message_filters.Subscriber('/joint_states', JointState)
    queue_size = 20 # how many sets of messages it should store from each input
                    # filter (by timestamp) while waiting for messages to arrive
                    # and complete their 'set'
    slop = 0.002# max delay (in seconds) that messages can still be sync
    optional_subs = []

    ball_sub=message_filters.Subscriber('/Ball/point', PointStamped)
    ts = message_filters.ApproximateTimeSynchronizer(
        [choppose_sub, ball_sub] , queue_size, slop)
    ts.registerCallback(callback)

def main():
    # set up storage subscriber
    global trajectory_count, datapoint_count, rosbag_name, start_timer
    init_subscriber()
    debug_counter = 1000000000000
    # start rosbag play

    for file in sorted(glob.glob(os.path.join(args.bag_folder, "*pose.bag"))):
        print(file)
        prefix_name = file[:-13]
        print("Working on {}".format(prefix_name))
        start_timer = None
        rosbag_name = os.path.basename(prefix_name)
        rosbag_player = start_rosbag_play(file)
        rosbag_player.communicate() # wait for rosbag play to finish
        print("-> accumulated {} datapoints".format(datapoint_count))
        debug_counter -= 1
        if debug_counter <= 0:
            break

    label_file.close()
    print("finished")

if __name__ == "__main__":
    write_csv_header(writer)
    main()
