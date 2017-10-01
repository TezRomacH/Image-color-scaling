# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import sys
import sklearn.preprocessing as pr
import matplotlib.pyplot as plt
from numpy import genfromtxt

BINARY_MODE = 'binary'
BIPOLAR_MODE = 'bipolar'
SCALE_MODE = 'scale'

SIGNAL_MODS = [BINARY_MODE, BIPOLAR_MODE]
MODS = [BINARY_MODE, BIPOLAR_MODE, SCALE_MODE]


def main():
    print(sys.argv)
    if len(sys.argv) >= 4:
        path = sys.argv[1]
        mode = sys.argv[2]
        assert mode in MODS, "Can't find mode \"{0}\"".format(mode)
        if mode in SIGNAL_MODS:
            data = np.array(Image.open(path))
            background = sys.argv[3].lstrip('#')
            c = int(sys.argv[4])  # границы
            color = [int(background[i:i + 2], 16) for i in (0, 2, 4)]
            reds = data[:, :, 0]
            greens = data[:, :, 1]
            blues = data[:, :, 2]
            mask = (reds >= color[0] - c) & (reds <= color[0] + c) & (greens >= color[1] - c) & (
            greens <= color[1] + c) & (blues >= color[2] - c) & (blues <= color[2] + c)

            image_copied = data.copy()
            image_copied[mask] = 0
            image_copied[~mask] = 255

            im = Image.fromarray(image_copied)
            im.save("outfile.jpg")

            plt.show(image_copied)

            signal_array = np.zeros(shape=(image_copied.shape[0], image_copied.shape[1]))
            signal_array[mask] = 1
            signal_array[~mask] = -1 if mode == BIPOLAR_MODE else 0

            # сохранение в файл
            np.savetxt(str(mode) + '_signal.csv', signal_array.flatten(), delimiter=',')
        elif mode == SCALE_MODE:
            (left, right) = (float(sys.argv[3]), float(sys.argv[4]))
            assert left < right, "Left corner should be less than right!"

            if "--file" in sys.argv:
                data = genfromtxt(path, delimiter=',')
                sum = data
                scaler = pr.MinMaxScaler(feature_range=(left, right))
                result = scaler.fit_transform(sum.reshape(-1, 1))
            else:
                data = np.array(Image.open(path))
                sum = np.asarray(data).sum(axis=2) / 3
                scaler = pr.MinMaxScaler(feature_range=(left, right))
                result = scaler.fit_transform(sum)
            np.savetxt('scale.csv', result.flatten(), delimiter=',')
    else:
        print(help())


def help():
    return """usage: path_to_file mode params
modes: binary, bipolar, scale
params:
    binary/bipolar: hexcolor, scalevalue
    scale: left, right [--file]
"""


main()  # path mode, hexcolor, c
