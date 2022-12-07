import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import print_message, read_dataset, \
    figsize_x, figsize_y, get_projection, chunks_dataset, chunks_size, \
    get_time_run_cum, subfolder_images, \
    annotation_forecast, annotation, annotation_run, options_savefig, \
    remove_collections, processes, truncate_colormap, home_folder
import sys
from computations import compute_rate
import pickle

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The one employed for the figure name when exported
variable_name = 'sat'

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
    dset = read_dataset(variables=['PMSL', 'SYNMSG_BT_CL_IR10.8'],
                        projection=projection)

    dset['prmsl'] = dset['prmsl'].metpy.convert_units('hPa').metpy.dequantify()
    dset['SYNMSG_BT_CL_IR10.8'] = dset['SYNMSG_BT_CL_IR10.8'].metpy.convert_units(
        'degC').metpy.dequantify()

    levels_clouds = np.arange(30, 100, 1)

    cmap_clouds = truncate_colormap(plt.get_cmap('Greys'), 0., 0.5)
    fp = open(home_folder + '/plotting/cmap_bt.pkl', 'rb')
    cmap_bt = pickle.load(fp)
    fp.close()

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()
    # Get coordinates from dataset
    m, x, y = get_projection(
        dset, projection, labels=True, color_borders='white')

    dset = dset.drop(['lon', 'lat']).load()

    levels_mslp = np.arange(dset['prmsl'].min().astype("int"),
                            dset['prmsl'].max().astype("int"), 4.)

    args = dict(x=x, y=y, ax=ax,
                levels_mslp=levels_mslp,
                levels_clouds=levels_clouds, time=dset.time,
                cmap_clouds=cmap_clouds, cmap_bt=cmap_bt)

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
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + \
            '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].pcolormesh(args['x'], args['y'],
                                   data['SYNMSG_BT_CL_IR10.8'],
                                   cmap=args['cmap_bt'],
                                   vmin=-73, vmax=22, antialiased=True, shading='auto')

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'],
                            'Satellite IR temperature',
                            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)

        if first:
            plt.colorbar(cs, orientation='horizontal',
                         label='Brightness temperature [C]', pad=0.03, fraction=0.03)

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
