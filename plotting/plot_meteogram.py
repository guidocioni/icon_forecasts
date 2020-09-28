debug = False
if not debug:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from glob import glob
import numpy as np
import pandas as pd
import os
from utils import *
import sys
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.ticker as ticker

print_message('Starting script to plot meteograms')

# Get the projection as system argument from the call so that we can
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message('City not defined, falling back to default (Hamburg)')
    cities = ['Hamburg']
else:
    cities = sys.argv[1:]

dset, time, cum_hour  = read_dataset(variables=['T_2M','TD_2M','T','VMAX_10M',
                                                'PMSL','HSURF','WW','RAIN_GSP','RAIN_CON',
                                                'SNOW_GSP','SNOW_CON','RELHUM','U','V'])
increments = (time[1:] - time[:-1]) / pd.Timedelta('1 hour')

for city in cities:  # This works regardless if cities is either single value or array
    print_message('Producing meteogram for %s' % city)
    lon, lat = get_city_coordinates(city)
    dset_city = dset.sel(lon=lon, lat=lat, method='nearest').load()
    dset_city['t'].metpy.convert_units('degC')
    dset_city['2t'].metpy.convert_units('degC')
    dset_city['2d'].metpy.convert_units('degC')
    dset_city['VMAX_10M'].metpy.convert_units('kph')
    dset_city['prmsl'].metpy.convert_units('hPa')
    dset_city['plev'].metpy.convert_units('hPa')

    rain_acc = dset_city['RAIN_GSP'] + dset_city['RAIN_CON']
    snow_acc = dset_city['SNOW_GSP'] + dset_city['SNOW_CON']
    rain = rain_acc.differentiate(coord="time", datetime_unit="h")
    snow = snow_acc.differentiate(coord="time", datetime_unit="h")

    weather_icons = get_weather_icons(dset_city['WW'], time)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])

    ax0 = plt.subplot(gs[0])
    cs = ax0.contourf(time, dset_city['t'].metpy.vertical.values, dset_city['t'].T, extend='both',
                      cmap=get_colormap("temp"), levels=np.arange(-70, 35, 5))
    ax0.axes.get_xaxis().set_ticklabels([])
    ax0.invert_yaxis()
    ax0.set_ylim(1000, 200)
    ax0.set_xlim(time[0], time[-1])
    ax0.set_ylabel('Pressure [Pa]')
    y_labels = ax0.get_yticks()
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cs2 = ax0.contour(time, dset_city['t'].metpy.vertical.values, dset_city['r'].T,
                      levels=np.linspace(0, 100, 5), colors='white', alpha=0.7)
    plt.clabel(cs2, fmt='%i', inline=True)
    dset_winds = dset_city.sel(time=pd.date_range(
        dset_city.time[0].values, dset_city.time[-1].values, freq='6H'))
    v = ax0.barbs(pd.to_datetime(dset_winds.time.values),
                  dset_winds['t'].metpy.vertical.values, dset_winds['u'].T, dset_winds['v'].T,
                  alpha=0.3, length=5.5)
    ax0.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax0.grid(True, alpha=0.5)
    an_fc = annotation_run(ax0, time)
    an_var = annotation(ax0, 'RH, $T$ and $u,v$ @(%3.1fN, %3.1fE, %d m)' % (dset_city.lat, dset_city.lon, dset_city.HSURF),
                        loc='upper left')
    an_city = annotation(ax0, city, loc='upper center')

    ax1 = plt.subplot(gs[1])
    ax1.set_xlim(time[0], time[-1])
    ts = ax1.plot(time, dset_city['2t'], label='2m $T$', color='darkcyan')
    ts1 = ax1.plot(
        time, dset_city['2d'], label='2m $T_d$', color='darkcyan', linestyle='dashed')
    ax1.axes.get_xaxis().set_ticklabels([])
    plt.legend(fontsize=7)
    ax1.set_ylabel('2m $T$, $T_d$ [$^{\circ}$C]')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.grid(True, alpha=0.5)

    for dt, weather_icon, dewp in zip(time, weather_icons, dset_city['2t']):
        imagebox = OffsetImage(weather_icon, zoom=.025)
        ab = AnnotationBbox(
            imagebox, (mdates.date2num(dt), dewp), frameon=False)
        ax1.add_artist(ab)

    ax2 = plt.subplot(gs[2])
    ax2.set_xlim(time[0], time[-1])
    ts = ax2.plot(time, dset_city['VMAX_10M'],
                  label='Gusts', color='lightcoral')
    ax2.set_ylabel('Wind gust [km/h]')
    ax22 = ax2.twinx()
    ts1 = ax22.plot(time, dset_city['prmsl'], label='MSLP', color='m')
    ax2.axes.get_xaxis().set_ticklabels([])
    ax22.set_ylabel('MSLP [hPa]')
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.grid(True, alpha=0.5)

    # Collect all the elements for the legend
    handles, labels = [], []
    for ax in (ax2, ax22):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    plt.legend(handles, labels, fontsize=7)

    ax3 = plt.subplot(gs[3])
    ax3.set_xlim(time[0], time[-1])
    ts = ax3.plot(time, rain_acc, label='Rain (acc.)',
                  color='dodgerblue', linewidth=0.1)
    ts1 = ax3.plot(time, snow_acc, label='Snow (acc.)',
                   color='orchid', linewidth=0.1)
    ax3.fill_between(time, rain_acc, y2=0, facecolor='dodgerblue', alpha=0.2)
    ax3.fill_between(time, snow_acc, y2=0, facecolor='orchid', alpha=0.2)
    ax3.set_ylim(bottom=0)
    ax3.set_ylabel('Accum. [mm]')
    ax33 = ax3.twinx()
    ts2 = ax33.plot(time, rain, label='Rain', color='dodgerblue')
    ts3 = ax33.plot(time, snow, label='Snow', color='orchid')
    ax33.set_ylim(bottom=0)
    ax33.set_ylabel('Inst. [mm h$^{-1}$]')
    ax33.legend(fontsize=7)

    ax3.grid(True, alpha=0.5)
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax3.xaxis.set_major_formatter(DateFormatter('%d %b %HZ'))
    for tick in ax3.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')

    fig.subplots_adjust(hspace=0.1)
    fig.colorbar(cs, orientation='vertical',
                 label='Temperature [C]', cax=cbar_ax)

    # Build the name of the output image
    filename = folder_images + '/meteogram_%s.png' % city

    if debug:
        plt.show(block=True)
    else:
        plt.savefig(filename, dpi=100, bbox_inches='tight')