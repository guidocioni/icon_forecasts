import numpy as np
from multiprocessing import Pool
from functools import partial
from utils import *
import sys
from computations import compute_snow_change

debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# The one employed for the figure name when exported 
variable_name = 'winter'

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
    dset = read_dataset(variables=['RAIN_GSP', 'RAIN_CON', 'H_SNOW', 'SNOWLMT'],
                        projection=projection)

    rain_acc = dset['RAIN_GSP'] + dset['RAIN_CON']
    rain = (rain_acc - rain_acc[0, :, :])
    rain = xr.DataArray(rain, name='rain_increment')

    dset.sde.metpy.convert_units('cm')
    dset = compute_snow_change(dset)

    dset = xr.merge([dset, rain])

    dset['SNOWLMT'].metpy.convert_units('m')

    levels_snow = (0.25, 0.5, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 150)
    levels_rain = (10, 15, 25, 35, 50, 75, 100, 125, 150)
    levels_snowlmt = np.arange(0., 3000., 500.)

    cmap_snow, norm_snow = get_colormap_norm("snow_wxcharts", levels_snow)
    cmap_rain, norm_rain = get_colormap_norm("rain", levels_rain)

    _ = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()

    m, x, y = get_projection(dset, projection, labels=True)
    m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500)

    dset = dset.drop(['RAIN_GSP', 'RAIN_CON']).load()

    # All the arguments that need to be passed to the plotting function
    args = dict(m=m, x=x, y=y, ax=ax,
             levels_snowlmt=levels_snowlmt, levels_rain=levels_rain,
             levels_snow=levels_snow, norm_snow=norm_snow,
             cmap_rain=cmap_rain, cmap_snow=cmap_snow, norm_rain=norm_rain)

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
        time, run, cum_hour = get_time_run_cum(data)
        # Build the name of the output image
        filename = subfolder_images[projection] + '/' + variable_name + '_%s.png' % cum_hour

        cs_rain = args['ax'].contourf(args['x'], args['y'], data['rain_increment'],
                         extend='max', cmap=args['cmap_rain'], norm=args['norm_rain'],
                         levels=args['levels_rain'], alpha=0.5, antialiased = True)
        cs_snow = args['ax'].contourf(args['x'], args['y'], data['snow_increment'],
                         extend='max', cmap=args['cmap_snow'], norm=args['norm_snow'],
                         levels=args['levels_snow'], antialiased = True)

        c = args['ax'].contour(args['x'], args['y'], data['SNOWLMT'], levels=args['levels_snowlmt'],
                             colors='red', linewidths=0.5)

        labels = args['ax'].clabel(c, c.levels, inline=True, fmt='%4.0f', fontsize=5)

        if projection != 'euratl':
            vals = add_vals_on_map(args['ax'],
                               projection,
                               data['snow_increment'].where(data['snow_increment'] >= 1),
                               args['levels_snow'],
                               cmap=args['cmap_snow'],
                               norm=args['norm_snow'],
                               density=5)

        an_fc = annotation_forecast(args['ax'], time)
        an_var = annotation(args['ax'], 'New snow and accumulated rain (since run start)',
            loc='lower left', fontsize=6)
        an_run = annotation_run(args['ax'], run)
        logo = add_logo_on_map(ax=args['ax'],
                                zoom=0.1, pos=(0.95, 0.08))

        if first:
            if projection == "euratl": 
                x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size     = 0.15, 0.15, 0.35, 0.02
                x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.15, 0.35, 0.02  
            elif projection == "de":
                x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size     = 0.17, 0.05, 0.32, 0.02
                x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.05, 0.32, 0.02 
            elif projection == "it":
                x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size     = 0.18, 0.05, 0.3, 0.02
                x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size = 0.55, 0.05, 0.3, 0.02 

            ax_cbar = plt.gcf().add_axes([x_cbar_0, y_cbar_0, x_cbar_size, y_cbar_size])
            ax_cbar_2 = plt.gcf().add_axes([x_cbar2_0, y_cbar2_0, x_cbar2_size, y_cbar2_size])
            cbar_snow = plt.gcf().colorbar(cs_snow, cax=ax_cbar, orientation='horizontal',
             label='Snow')
            cbar_rain = plt.gcf().colorbar(cs_rain, cax=ax_cbar_2, orientation='horizontal',
             label='Rain')
            cbar_snow.ax.tick_params(labelsize=8) 
            cbar_rain.ax.tick_params(labelsize=8)

        if debug:
            plt.show(block=True)
        else:
            plt.savefig(filename, **options_savefig)

        remove_collections([cs_rain, cs_snow, c, labels, an_fc, an_var, an_run, logo])
        if projection != 'euratl':
            remove_collections([vals])

        first = False 


if __name__ == "__main__":
    import time
    start_time=time.time()
    main()
    elapsed_time=time.time()-start_time
    print_message("script took " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

