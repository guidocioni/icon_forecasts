import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import print_message, read_dataset, \
    figsize_x, figsize_y, get_projection, chunks_dataset, chunks_size, \
    get_time_run_cum, subfolder_images, \
    annotation_forecast, annotation, annotation_run, options_savefig, \
    remove_collections, processes, get_colormap_norm
import sys

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The one employed for the figure name when exported
variable_name = 'precip_acc_24'

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
    dset = read_dataset(variables=['TOT_PREC'],
                        projection=projection)

    levels_precip = list(np.arange(1, 50, 0.4)) + \
        list(np.arange(51, 100, 2)) +\
        list(np.arange(101, 200, 3)) +\
        list(np.arange(201, 500, 6)) + \
        list(np.arange(501, 1000, 50)) + \
        list(np.arange(1001, 2000, 100))

    dset = dset.resample(time="24H",
                         base=dset.time[0].dt.hour.item()).nearest().diff(dim='time')

    cmap, norm = get_colormap_norm('rain_acc_wxcharts', levels=levels_precip)

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)
    # additional maps adjustment for this map
    m.arcgisimage(service='World_Shaded_Relief', xpixels=1500)

    dset = dset.drop(['lon', 'lat']).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(x=x, y=y, ax=ax,
                levels_precip=levels_precip,
                cmap=cmap, norm=norm)

    print_message('Pre-processing finished, launching plotting scripts')
    if debug:
        plot_files(dset.isel(time=slice(-2, -1)), **args)
    else:
        # Parallelize the plotting by dividing into chunks and processes
        dss = chunks_dataset(dset, chunks_size)
        plot_files_param = partial(plot_files, **args)
        p = Pool(processes)
        p.map(plot_files_param, dss)


def plot_files(dss, **args):
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + \
            '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['tp'],
                                 extend='max',
                                 cmap=args['cmap'],
                                 norm=args['norm'],
                                 levels=args['levels_precip'])

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'Accumulated precipitation in the last 24 hours',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)

        if first:
            plt.colorbar(cs, orientation='horizontal', label='Accumulated precipitation [mm]',
                         pad=0.035, fraction=0.04)

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
