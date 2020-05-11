from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def drawxyz(ax, center, x, z):


def main():
    print('hello')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = [[0,1],[0,0],[0,0]]
    # ax.arrow(0,0,0,1,length_includes_head=True,head_width=0.25, head_length=0.5, fc='r', ec='b')
    theta = [0,0,0,0,0,0]
    joint = [[0,0,0]]*6
    point0 = [0,0,0]
    
    ax.quiver(0, 0, 0, 1, 1, 1, length = 0.5, normalize = True)
    ax.set_xlim3d(0, 0.8)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(0, 1.0)


    

    plt.show()

if __name__ == "__main__":
    main()