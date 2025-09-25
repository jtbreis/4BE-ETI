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
box_size_initial_y = 2
# box size in z direction for track initialization
box_size_initial_z = 2
# box size used after a track is initialized (this should be as small as possible to
# eliminate spurious track)
box_size = 1


"""
Start of the code; user should not change anything after this point 
"""


# data = basic_utils.load_data(main_dir, trial_name, file_type)
data = basic_utils.load_data_h5(os.path.join(folder, filename))

logger = basic_utils.initialize_logfile(
    folder, 'logfile_' + run + '.log')

logging.info(folder)
logging.info(run)
logging.info('Particle tracking initialized.')


start_time = time.time()

for jj in range(0, 3):
    if jj % 500 == 0:
        logging.info(str(jj))
        logging.info(str(time.time() - start_time) + 'seconds')
    imInit = np.where(data.Slice == jj)[0]
    for ii in range(len(imInit)):
        if data.Count[imInit[ii]] == 0:
            if dimension == '3d':
                data = no_previous_tracks_3d(data, jj, ii, box_size,
                                             box_size_initial_x,
                                             box_size_initial_y,
                                             box_size_initial_z)
            else:
                data = no_previous_tracks(data, jj, ii, box_size,
                                          box_size_initial_x,
                                          box_size_initial_y)
        else:
            if dimension == '3d':
                data = previous_tracks_3d(data, jj, ii, box_size)
            else:
                data = previous_tracks(data, jj, ii, box_size)

    if len(data.Count[data.Count == -1]) > 0:
        data.Count[data.Count == (-1)] = 0

if dimension == '3d':
    data_final = pd.DataFrame({'X': data.x, 'Y': data.y, 'Z': data.z,
                               'Slice': data.Slice, 'Count': data.Count,
                               'Cost': data.Cost, 'Area': data.Area})
else:
    data_final = pd.DataFrame({'X': data.x, 'Y': data.y,
                               'Slice': data.Slice, 'Count': data.Count,
                               'Cost': data.Cost, 'Area': data.Area})

# data_final.to_pickle(
#     folder + 'tracking_results_box_size_' + run + '.pkl')

data_final.to_hdf(
    os.path.join(folder, f'tracking_results_box_size_{run}.h5'),
    key='df', mode='w'
)

logging.info('Particle tracking program took ' +
             str(time.time() - start_time) + ' seconds to run.')
