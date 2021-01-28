import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported 
variable_name = 'hsnow_anom'

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
    dset = read_dataset(variables=['H_SNOW'],
                        projection=projection)

    original_time = dset.time
    run = dset['run']
    # Mean over day of the year
    dset = dset.groupby(dset.time.dt.dayofyear).mean()
    # Read climatology remapped over ICON-EU grid
    clima = xr.open_dataset('/home/ekman/guido/climatologies/clima_1981-2010_uerra_snow_remap_iconeu.nc').squeeze().sel(time='2010')
    # Also year transform time to dayoftheyear to compare 
    clima = clima.rename({'time': 'dayofyear'}).assign_coords({'dayofyear': clima.time.dt.dayofyear.values})
    # merge the two datasets
    merged = xr.merge([clima, dset], join='inner')
    # now compute anomaly
    merged['sde'] = merged['sde'] * 100.
    merged['anomaly'] = merged['sde'] - merged['sd']
    # Transform back the time dimension to the "forecast" time with the first input 
    merged = merged.rename({'dayofyear': 'time'}).assign_coords({'time': original_time.resample(time='1D').first()})
    # Conver the clima
    merged['run'] = run

    levels_temp = np.arange(-100, 101, 5)

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(merged, projection, labels=True)

    merged = merged.drop(['lon', 'lat']).load()


    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax, levels_temp=levels_temp)

    print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(merged.isel(time=slice(0, 2)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and processes 
        dss = chunks_dataset(merged, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(processes)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for i, time_sel in enumerate(dss.time):
        data = dss.sel(time=time_sel)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % i

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['anomaly'],
                                 extend='both', 
                                 cmap='BrBG',
                                 levels=args['levels_temp'])


        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Daily snow thickness anomaly (w.r.t to UERRA 1981-2010 clima)',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Anomaly (cm)', pad=0.035, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs, an_fc, an_var, an_run, logo])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

