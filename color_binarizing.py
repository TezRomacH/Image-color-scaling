# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

BINARY_MODE = 'binary'
BIPOLAR_MODE = 'bipolar'
MODS = [BINARY_MODE, BIPOLAR_MODE]


def parse_args():
    program_examples = '''Example of use:
    python color_binarizing.py --path=test.jpg --mode=bipolar F3AA19
    python color_binarizing.py -p=img.png --mode=binary ff0000 -r=50
    python color_binarizing.py --path=cat.jpg -m=binary ff0000 -o=bin_cat.png --radius=100
    '''

    parser = argparse.ArgumentParser(
        description='Utility for binarize images with vectorizing it to float interval',
        epilog=program_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--path', type=str, help='Path to file (image or csv)', required=True)
    parser.add_argument('-m', '--mode', type=str, choices=MODS, default=BINARY_MODE, help='Mode to images')
    parser.add_argument('background', help='Hexcolor of background')
    parser.add_argument('-r', '--radius', metavar='COLOR_RAD', type=int, help='Radius of color sphere', default=30)
    parser.add_argument('-o', '--out', default="outfile.jpg", metavar='OUTPUT', help='Path to output image')
    return parser.parse_args()


def main():
    args_res = parse_args()
    path = args_res.path
    mode = args_res.mode

    radius = args_res.radius
    color = parse_hexcolor(args_res.background)

    data = np.array(Image.open(path))
    if len(data.shape) > 2:
        reds = data[:, :, 0]
        greens = data[:, :, 1]
        blues = data[:, :, 2]
    else:
        reds = data[:, :]
        greens = data[:, :]
        blues = data[:, :]
    # find pixels near to background
    mask = (reds >= color[0] - radius) & (reds <= color[0] + radius) & (greens >= color[1] - radius) & (
        greens <= color[1] + radius) & (blues >= color[2] - radius) & (blues <= color[2] + radius)

    # binarize copy of image by mask
    image_copied = data.copy()
    image_copied[mask] = 0
    image_copied[~mask] = 255

    im = Image.fromarray(image_copied)
    im.save(args_res.out)

    plt.show(image_copied)

    # fill our signal vector of values depends of current mode
    signal_vector = np.zeros(shape=(image_copied.shape[0], image_copied.shape[1]))
    signal_vector[mask] = 1
    signal_vector[~mask] = -1 if mode == BIPOLAR_MODE else 0

    np.savetxt(str(mode) + '_vector.csv', signal_vector.flatten(), delimiter=',')


def parse_hexcolor(color):
    return [int(color[i:i + 2], 16) for i in (0, 2, 4)]


main()
