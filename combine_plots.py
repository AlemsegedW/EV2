import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys


def combine_plots():
    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.imshow(mpimg.imread('user_cs.png'))
    ax1.axis('off')
    ax2.imshow(mpimg.imread('user_cs.png'))
    ax2.axis('off')
    ax3.imshow(mpimg.imread('user_cs.png'))
    ax3.axis('off')
    plt.savefig('user_cs_combined')
    plt.show(block=False)
    plt.close('user_cs_combined')


if __name__ == "__main__":
    combine_plots()
