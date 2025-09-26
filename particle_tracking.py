import basic_utils
import logging
import os
import time

import numpy as np
import pandas as pd

from particle_tracking_code import previous_tracks
from particle_tracking_code import previous_tracks_3d
from particle_tracking_code import no_previous_tracks
from particle_tracking_code import no_previous_tracks_3d


"""
Runs particle tracking code on a given file

Required user inputs are directly below
"""

# directory containing the file with particle information (location, area, frame number)
# you must include the full path to the directory here
folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_aligned_with_gravity/Run1/'
filename = 'rays_out_cpp.h5'
run = 'run1'

# 2d or 3d tracking?
dimension = '3d'
# box size in x direction for track initialization (a good initial guess is the expected
# maximum displacement of the particles in the x direction between frames)
box_size_initial_x = 2
# box size in y direction for track initialization
box_size_initial_y = 1.5
# box size in z direction for track initialization
box_size_initial_z = 1
# box size used after a track is initialized (this should be as small as possible to
# eliminate spurious track)
box_size = 1
track_length = 4


"""
Start of the code; user should not change anything after this point
"""

logger = basic_utils.initialize_logfile(
    folder, 'logfile_' + run + '.log')

logging.info(folder)
logging.info(run)
logging.info('Particle tracking initialized.')


start_time = time.time()
# Create an array of frame index ranges, each containing 4 consecutive frames
# assuming Slice contains frame indices starting from 0
num_frames = basic_utils.get_nframes(os.path.join(folder, filename))
frame_ranges = [list(range(i, min(i + track_length, num_frames)))
                for i in range(0, num_frames, track_length)]

basic_utils.create_h5_file(folder=folder)

for frame_range in frame_ranges:
    print(f'Processing frames {frame_range}')
    data_range = basic_utils.load_data_h5(
        os.path.join(folder, filename), frame_range)
    for jj in range(min(frame_range), max(frame_range)):
        if jj % 500 == 0:
            logging.info(str(jj))
            logging.info(str(time.time() - start_time) + 'seconds')
        imInit = np.where(data_range.Slice == jj)[0]
        for ii in range(len(imInit)):
            if data_range.Count[imInit[ii]] == 0:
                if dimension == '3d':
                    data_range = no_previous_tracks_3d(data_range, jj, ii, box_size,
                                                       box_size_initial_x,
                                                       box_size_initial_y,
                                                       box_size_initial_z)
                else:
                    data_range = no_previous_tracks(data_range, jj, ii, box_size,
                                                    box_size_initial_x,
                                                    box_size_initial_y)
            else:
                if dimension == '3d':
                    data_range = previous_tracks_3d(
                        data_range, jj, ii, box_size)
                else:
                    data_range = previous_tracks(data_range, jj, ii, box_size)

        if len(data_range.Count[data_range.Count == -1]) > 0:
            data_range.Count[data_range.Count == (-1)] = 0

    basic_utils.write_data_h5(data_range, folder, frame_range)

logging.info('Particle tracking program took ' +
             str(time.time() - start_time) + ' seconds to run.')
