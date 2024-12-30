import netCDF4 as nc

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords, extract_vars)

import numpy as np
from pathlib import Path

#Dimensions are (Batch size, channels, frames, depth, length, width)
#shape is (B, 19, F, 47, 265, 442)



def extract_data(path, seq_length):
    features = ["T"]  #"PM2_5_DRY", "co", "no2", "o3", "so2", "PM10", "QVAPOR", "T", "P", "U", "V",
                #"W"# long, lat, z, 4 seasons

    dataset = []
    frames = []
    frame_num = 0
    files = sorted(Path(path).glob("*"))
    for file in files:
        x = []
        print(file.stem)
        data = nc.Dataset(file)

        helper_x = []
        helper_y = []
        helper_z = np.zeros((47, 265, 442))

        temp_x = getvar(data, "XLONG")
        temp_y = getvar(data, "XLAT")

        for i in range(0, 47):
            for j in range(0, 265):
                for k in range(0, 442):
                    helper_z[i][j][k] = i

            helper_x.append(temp_x)
            helper_y.append(temp_y)

        for feature in features:
            if feature == "U":
                x.append(getvar(data, feature)[:, :, :-1])
            elif feature == "V":
                x.append(getvar(data, feature)[:, :-1])
            elif feature == "W":
                x.append(getvar(data, feature)[:-1])
            else:
                x.append(getvar(data, feature))

        x.append(helper_x)
        x.append(helper_y)
        x.append(helper_z)

        name = Path(file).stem
        name = name.split("-")

        # add seasons
        if 12 == int(name[1]) or 1 == int(name[1]) or 2 == int(name[1]):
            x.append(np.ones((47, 265, 442)))
        else:
            x.append(np.zeros((47, 265, 442)))

        if 3 == int(name[1]) or 4 == int(name[1]) or 5 == int(name[1]):
            x.append(np.ones((47, 265, 442)))
        else:
            x.append(np.zeros((47, 265, 442)))

        if 6 == int(name[1]) or 7 == int(name[1]) or 8 == int(name[1]):

            x.append(np.ones((47, 265, 442)))
        else:
            x.append(np.zeros((47, 265, 442)))

        if 9 == int(name[1]) or 10 == int(name[1]) or 11 == int(name[1]):

            x.append(np.ones((47, 265, 442)))
        else:
            x.append(np.zeros((47, 265, 442)))

        frames.append(x)
        frame_num += 1
        if frame_num == seq_length:
            frame_num = 0
            dataset.append(frames)
            frames = []

        print(np.shape(dataset))

    return dataset


#
fp= r"D:\files2\train"
train = extract_data(fp,3)
train_swap=np.swapaxes(train, 1, 2)
print(train_swap.shape)
np.save(r"D:\files2\output\anvesh", train_swap)

