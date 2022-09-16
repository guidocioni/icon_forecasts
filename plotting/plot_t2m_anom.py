import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import print_message, read_dataset, \
    figsize_x, figsize_y, get_projection, chunks_dataset, chunks_size, \
    get_time_run_cum, subfolder_images, \
    annotation_forecast, annotation, annotation_run, options_savefig, \
    remove_collections, processes
import xarray as xr
import sys
# from matplotlib import patheffects
# import metpy.calc as mpcalc

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The one employed for the figure name when exported
variable_name = 't_2m_anom'

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
    dset = read_dataset(variables=['T_2M'],
                        projection=projection)

    original_time = dset.time
    run = dset['run']
    # Mean over day of the year
    dset = dset.groupby(dset.time.dt.dayofyear).mean()
    # Read climatology remapped over ICON-EU grid
    clima = xr.open_dataset(
        '/home/ekman/guido/climatologies/clima_1981-2010_mescan_t2m_mean_remap_iconeu.nc').squeeze()
    # Also year transform time to dayoftheyear to compare
    clima = clima.rename({'time': 'dayofyear'}).assign_coords(
        {'dayofyear': clima.time.dt.dayofyear.values})
    # remove duplicates in time
    clima = clima.sel(dayofyear=~clima.get_index("dayofyear").duplicated())
    # merge the two datasets
    merged = xr.merge([clima.rename({'2t': '2t_clim'}), dset], join='inner')
    # now compute anomaly
    merged['anomaly'] = merged['2t'] - merged['2t_clim']
    # Transform back the time dimension to the "forecast" time with the first input
    merged = merged.rename({'dayofyear': 'time'}).assign_coords(
        {'time': original_time.resample(time='1D').first()})
    # Conver the clima
    merged['2t_clim'] = merged['2t_clim'] - 273.15
    merged['2t'] = merged['2t'] - 273.15
    merged['run'] = run

    levels_temp = np.arange(-20, 21)

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)

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
        # data['t_clima'].values = mpcalc.smooth_n_point(data['t_clima'].values, n=9, passes=9)
        # Build the name of the output image
        filename = subfolder_images[projection] + \
            '/' + variable_name + '_%s.png' % i

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['anomaly'],
                                 extend='both',
                                 cmap='seismic',
                                 levels=args['levels_temp'])

#         css = args['ax'].contour(args['x'], args['y'],
#                                data['2t'],
#                                levels=np.arange(-25., 40., 3.),
#                                colors='gray', linewidths=0.5,
#                               linestyles='solid')

#         labels2 = args['ax'].clabel(
#             css, css.levels, inline=True, fmt='%4.0f', fontsize=8, zorder=10)
#         plt.setp(labels2, path_effects=[
#         patheffects.withStroke(linewidth=0.1, foreground="green")])

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Daily 2m temp. anomaly (w.r.t to MESCAN-SURFEX 1981-2010 clima)',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)

        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='Anomaly (°C)', pad=0.035, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs, an_fc, an_var, an_run])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S",
                  time.gmtime(elapsed_time)))
