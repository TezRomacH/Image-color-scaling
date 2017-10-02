# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import sys
import sklearn.preprocessing as pr
from numpy import genfromtxt
import argparse

program_examples = '''Example of use:
    python color_scaling.py --path=test.csv -1.5 1.5 --file
    python color_scaling.py -p="animals/1.jpg" -90 90
    python color_scaling.py --path="img.png" --out="output.csv"
'''

parser = argparse.ArgumentParser(
    description='Utility for scale image/csv to float interval',
    epilog=program_examples,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-p', '--path', type=str, help='Path to file (image or csv)', required=True)
parser.add_argument('left', type=float, nargs='?', help='Left edge of scaling', default=-1.0)
parser.add_argument('right', type=float, nargs='?', help='Right edge of scaling', default=1.0)
parser.add_argument('-f', '--file', action='store_true', help='Flag to use csv instead of images')
parser.add_argument('-o', '--out', nargs='?', default="scale.csv", help='Path to output file')
args_res = parser.parse_args()


def main():
    path = args_res.path
    (left, right) = (args_res.left, args_res.right)
    if left > right:
        print("Error! Left edge should be less than right!")
        parser.print_usage()
        sys.exit(2)
    scaler = pr.MinMaxScaler(feature_range=(left, right))

    if args_res.file:
        # read csv file
        data = genfromtxt(path, delimiter=',')
        result = scaler.fit_transform(data.reshape(-1, 1))
    else:
        data = np.array(Image.open(path))
        # Mean value of RGB by each row
        mean_pixels = np.asarray(data).sum(axis=2) / 3
        result = scaler.fit_transform(mean_pixels)
    np.savetxt(args_res.out, result.flatten(), delimiter=',')


main()
