__author__      = "Nancy Xin Ru Wang"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.animation import FuncAnimation
import pdb
import pandas as pd
from datetime import timedelta, datetime
from ast import literal_eval
from scipy.signal import savgol_filter
import argparse

def plotSkeleton(data, opts, handle, thresh, i, start):
    j, c = data[0], data[1]
    if len(opts) == 0:
        opts = plotSkeletonDefaultopts(opts)

    joints = range(7)
    # draw skeleton
    clr = 8
    if np.any(c[[4, 6]] < thresh):
        clr += 2
    handle[0].set_data(j[0, [4, 6]], 480-j[1, [4, 6]])
    handle[0].set_color(opts["clr"][clr])
    clr = 8
    if np.any(c[[3, 5]] < thresh):
        clr += 2
    handle[1].set_data(j[0, [3, 5]], 480-j[1, [3, 5]])
    handle[1].set_color(opts["clr"][clr])
    clr = 9
    if np.any(c[[2, 4]] < thresh):
        clr += 2
    handle[2].set_data(j[0, [2, 4]], 480-j[1, [2, 4]])
    handle[2].set_color(opts["clr"][clr])
    clr = 9
    if np.any(c[[1, 3]] < thresh):
        clr += 2
    handle[3].set_data(j[0, [1, 3]], 480-j[1, [1, 3]])
    handle[3].set_color(opts["clr"][clr])
    for j2 in joints:
        handle[j2+4].set_offsets([[j[0, j2],480-j[1, j2]]])
    handle[-1].set_text(str(start + timedelta(seconds=i/30.0))[11:19])
    return handle

def plotSkeletonDefaultopts(opts):

    opts["clr"] = [cmx.jet(x)[1:] for x in xrange(12)]
    #sets coulour of joints
    opts["clr"][8] = (1,0,0)
    opts["clr"][9] = (0,1,0)
    opts["clr"][10] = (1,0,1)
    opts["clr"][11] = (0,1,1)
    opts["linewidth"] = 2
    opts["jointsize"] = 6

    if 'jointlinewidth' not in opts.keys():
        opts["jointlinewidth"] = 1
    if 'jointlinecolor' not in opts.keys():
        opts["jointlinecolor"] = np.zeros(shape=(7, 3))
    if not hasattr(opts["jointsize"], "__len__"):
        opts["jointsize"] = opts["jointsize"] * np.zeros(shape=(7, 1)) + 10
        opts["jointsize"][0] = 60

    if not hasattr(opts["jointlinewidth"], "__len__"):
        opts["jointlinewidth"] = opts["jointlinewidth"] * np.zeros(shape=(7, 1)) + 1

    return opts

def init():
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    for obj in handle[:4]:
        obj.set_data([],[])
    for obj in handle[4:-1]:
        obj.set_offsets([])
    return handle

def update(i):
    return plotSkeleton([joints_to_plot[0][i].reshape(2,7), joints_to_plot[1][i]], opts, handle, thresh, i, start)


def visualize(joints, confidence):
    plotSkeleton(joints, confidence, {}, {}, ax, True)
    plt.show()


def convert_joints(row):
    row_processed = [literal_eval(row[col]) for col in ["head", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist"]]
    joint_loc = np.vstack([row_processed[i][:2] for i in range(len(row_processed))]).T
    confidence = np.array([row_processed[i][2] for i in range(len(row_processed))])
    return [joint_loc, confidence]

def load_period(df, start, duration, smooth=False):
    df_sub = df.loc[(df.time>start) & (df.time <(start + timedelta(seconds=duration)))]
    result = [convert_joints(row) for index, row in df_sub.iterrows()]
    coords = np.vstack(np.vstack(result)[:,0]).reshape(len(result), 14)
    conf = np.vstack(result)[:,1]
    if smooth:
        coords = savgol_filter(coords, 11,3, axis=0)
    return [coords, conf]

def date_parser(string_list):
    return datetime.strptime("01-01-1000 " + string_list,
                      '%m-%d-%Y %H:%M:%S.%f')


# Global params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help="File location of joint location csv")
    parser.add_argument('-s', '--start', type=str, required=True, help="Start time to animate, in the format of HH:MM:SS.IIIIII like 11:55:07.000000")
    parser.add_argument('-d', '--dur', type=float, required=True, help="Duration to animate, in seconds")
    parser.add_argument('-th', '--thresh', type=float, default=0.25, help="Confidence threshold")
    parser.add_argument('-sm', '--smooth', type=float, default=0, help="0: No smoothing 1: Temporal smoothing")
    args = parser.parse_args()

    df = pd.read_csv(args.file, parse_dates=[1], date_parser=date_parser)
    input_start = args.start
    input_dur = args.dur
    start = datetime.strptime("01-01-1000 " + input_start,
                              '%m-%d-%Y %H:%M:%S.%f')

    joints_to_plot = load_period(df, start, input_dur, smooth=args.smooth)
    opts = plotSkeletonDefaultopts({})
    fig, ax = plt.subplots()
    handle = []
    for i in xrange(4):
        lobj = ax.plot([], [], linewidth=opts["linewidth"])[0]
        handle.append(lobj)
    for i in xrange(7):
        lobj = ax.scatter([], [], s=opts["jointsize"][i])
        handle.append(lobj)
    handle.append(plt.text(0.5, 0.5, ""))
    handle = tuple(handle)
    thresh = args.thresh
    ani = FuncAnimation(fig, update, frames=len(joints_to_plot[1]),
                        init_func=init, blit=True, interval=1 / 30.0 * 1000)
    plt.show()


