import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys
from matplotlib import patheffects
import metpy.calc as mpcalc

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported 
variable_name = 'tmax_2m_anom'

print_message('Starting script to plot '+variable_name)

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message(
        'Projection not defined, falling back to default (euratl)')
    projection = 'euratl'
else:
    projection = sys.argv[1]


def main():
    """In the main function we basically read the files and prepare the variables to be plotted.
    This is not included in utils.py as it can change from case to case."""
    dset = read_dataset(variables=['TMAX_2M'], projection=projection)

    original_time = dset.time
    run = dset['run']
    # Mean over day of the year
    dset = dset.groupby(dset.time.dt.dayofyear).max()
    # Read climatology remapped over ICON-EU grid
    clima = xr.open_dataset('/home/ekman/guido/climatologies/clima_1997-2019_cosmo_rea6_surface_variables_remap_iconeu.nc').squeeze()['mx2t6']
    # Also year transform time to dayoftheyear to compare 
    clima = clima.rename({'time': 'dayofyear'}).assign_coords({'dayofyear': clima.time.dt.dayofyear.values})
    # remove duplicates in time
    clima = clima.sel(dayofyear=~clima.get_index("dayofyear").duplicated())
    # merge the two datasets
    merged = xr.merge([clima, dset], join='inner')
    # now compute anomaly
    merged['anomaly'] = merged['TMAX_2M'] - merged['mx2t6']
    # Transform back the time dimension to the "forecast" time with the first input 
    merged = merged.rename({'dayofyear': 'time'}).assign_coords({'time': original_time.resample(time='1D').first()})
    # Conver the clima 
    merged['TMAX_2M'] = merged['TMAX_2M'] - 273.15
    merged['mx2t6'] = merged['mx2t6'] - 273.15
    merged['run'] = run


    levels_temp = np.arange(-20, 21)

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)

    merged = merged.load()

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_temp=levels_temp)

    print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(merged.isel(time=slice(0, 2)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and processes 
        dss = chunks_dataset(merged, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(6)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for i, time_sel in enumerate(dss.time):
        data = dss.sel(time=time_sel)
        time, run, cum_hour = get_time_run_cum(data)
        #data['t_clima'].values = mpcalc.smooth_n_point(data['t_clima'].values, n=9, passes=9)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % i

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['anomaly'],
                                 extend='both',
                                 cmap='seismic',
                                 levels=args['levels_temp'])

        # plot every -th element
        if projection == "euratl":
            density = 28
        elif projection == "it":
            density = 6
        elif projection == "de":
            density = 5

        vals = add_vals_on_map(args['ax'], projection,
                               data['TMAX_2M'], np.arange(-25, 40, 1),
                               cmap = get_colormap("temp"),
                               fontsize=7,
                               density=density)

        for val in vals:
            val.set_alpha(0.4)

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Daily 2m Maximum temp. anomaly (w.r.t to COSMO-REA6 1997-2018 clima) with forecast (values)',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Anomaly (°C)', pad=0.035, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs, an_fc, an_var, an_run, vals, logo])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

