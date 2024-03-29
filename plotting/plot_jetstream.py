import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import plot_maxmin_points, print_message, read_dataset, \
    figsize_x, figsize_y, get_projection, chunks_dataset, chunks_size, \
    get_time_run_cum, subfolder_images, \
    annotation_forecast, annotation, annotation_run, options_savefig, \
    remove_collections, processes, truncate_colormap
import sys
from computations import compute_geopot_height, compute_wind_speed

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The one employed for the figure name when exported
variable_name = 'winds_jet'

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
    dset = read_dataset(variables=['U', 'V', 'FI'],
                        level=30000,
                        projection=projection)
    dset = compute_wind_speed(dset)
    dset = compute_geopot_height(dset)

    levels_wind = np.arange(60., 300., 10.)
    levels_gph = np.arange(8200., 9700., 80.)

    cmap = truncate_colormap(plt.get_cmap('CMRmap_r'), 0., 0.9)

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500)

    dset = dset.drop(['lon', 'lat', 'u', 'v', 'z']).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax,
                levels_wind=levels_wind,
                levels_gph=levels_gph, time=dset.time,
                cmap=cmap)

    print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(dset.isel(time=slice(0, 2)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and processes
        dss = chunks_dataset(dset, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(processes)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + \
            '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['wind_speed'],
                                 extend='max',
                                 cmap=args['cmap'],
                                 levels=args['levels_wind'])

        c = args['ax'].contour(args['x'], args['y'],
                               data['geop'],
                               levels=args['levels_gph'],
                               colors='black', linewidths=0.5)

        labels = args['ax'].clabel(
            c, c.levels, inline=True, fmt='%4.0f', fontsize=6)

        maxlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], data['geop'],
                                       'max', 150, symbol='H', color='royalblue', random=True)
        minlabels = plot_maxmin_points(args['ax'], args['x'], args['y'], data['geop'],
                                       'min', 150, symbol='L', color='coral', random=True)

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Winds and geopotential [m] @300hPa',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)

        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='Wind [km/h]', pad=0.035, fraction=0.04)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([c, cs, labels, an_fc, an_var,
                           an_run, maxlabels, minlabels])

        first = False


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed_time = time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S",
                  time.gmtime(elapsed_time)))
