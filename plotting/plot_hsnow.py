import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys
from computations import compute_snow_change
import metpy.calc as mpcalc

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import seaborn as sns

# The one employed for the figure name when exported 
variable_name = 'hsnow'

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
    dset = read_dataset(variables=['H_SNOW', 'SNOWLMT'],
                        projection=projection)
    dset['sde'].metpy.convert_units('cm')
    dset['SNOWLMT'].metpy.convert_units('m')

    dset = compute_snow_change(dset)

    levels_hsnow = (-50, -40, -30, -20, -10, -5, -2.5, -2, -1, -0.5,
                    0, 0.5, 1, 2, 2.5, 5, 10, 20, 30, 40, 50)
    levels_snowlmt = np.arange(0., 3000., 500.)

    cmap, norm = from_levels_and_colors(levels_hsnow, 
                                        sns.color_palette("PuOr", 
                                                          n_colors=len(levels_hsnow) + 1),
                                        extend='both')

    _ = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()        
    # Get coordinates from dataset
    m, x, y = get_projection(dset, projection, labels=True)
    #m.fillcontinents(color='lightgray',lake_color='whitesmoke', zorder=0)
    m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500)

    dset = dset.drop(['lon', 'lat', 'sde']).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(m=m, x=x, y=y, ax=ax, cmap=cmap, norm=norm,
                 levels_hsnow=levels_hsnow,
                 levels_snowlmt=levels_snowlmt, time=dset.time)


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
    # Using args we don't have to change the prototype function if we want to add other parameters!
    first = True
    for time_sel in dss.time:
        data = dss.sel(time=time_sel)
        #data['SNOWLMT'].values = mpcalc.smooth_n_point(data['SNOWLMT'].values, n=5, passes=4)
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % cum_hour

        cs = args['ax'].contourf(args['x'], args['y'],
                                 data['snow_increment'],
                                 extend='both', 
                                 cmap=args['cmap'],
                                 norm=args['norm'],
                                 levels=args['levels_hsnow'])

        css = args['ax'].contour(args['x'], args['y'],
                                 data['snow_increment'],
                                 levels=args['levels_hsnow'],
                                 colors='gray',
                                 linewidths=0.2)

        labels2 = args['ax'].clabel(css, css.levels,
            inline=True, fmt='%4.0f', fontsize=6)

        c = args['ax'].contour(args['x'], args['y'],
                               data['SNOWLMT'],
                               levels=args['levels_snowlmt'],
                               colors='red', linewidths=0.5)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f' , fontsize=5)

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'],
            'Snow depth change [cm] since run beginning and snow limit [m]',
            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            cb = plt.colorbar(cs, orientation='horizontal', label='Snow depth change [m]',
                pad=0.038, fraction=0.035, ticks=args['levels_hsnow'][::2])
            cb.ax.tick_params(labelsize=7)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)        

        remove_collections([c, cs, css, labels, labels2, an_fc, an_var, an_run, logo])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

