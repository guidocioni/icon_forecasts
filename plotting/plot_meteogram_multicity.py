debug = True
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

print_message('Starting script to plot meteograms')

# Get the projection as system argument from the call so that we can 
# span multiple instances of this script outside
if not sys.argv[1:]:
    print_message('City not defined, falling back to default (Hamburg)')
    cities = ['Hamburg']
else:    
    cities=sys.argv[1:]

file = glob(input_file)
print_message('Using file '+file[0])
dset = xr.open_dataset(file[0])
dset = dset.metpy.parse_cf()
hsurf = xr.open_dataset('/home/mpim/m300382/icon_forecasts/ICON_EU_invariant.nc')['HSURF']

time = pd.to_datetime(dset.time.values)
increments = (time[1:] - time[:-1]) / pd.Timedelta('1 hour') 
cum_hour=np.array((time-time[0]) / pd.Timedelta('1 hour')).astype("int")

fig = plt.figure(figsize=(10, 2*len(cities)))

gs = gridspec.GridSpec(len(cities), 1) 
i=0

for city in cities:# This works regardless if cities is either single value or array
    lon, lat = get_city_coordinates(city)
    dset_city =  dset.sel(lon=lon, lat=lat, method='nearest')
    height = hsurf.sel(lon=lon, lat=lat, method='nearest')
    dset_city['2t'] = dset_city['2t'].metpy.convert_units('degC').metpy.dequantify()

    rain_acc = dset_city['RAIN_GSP'] + dset_city['RAIN_CON']
    snow_acc = dset_city['SNOW_GSP'] + dset_city['SNOW_CON']
    rain = rain_acc.diff(dim='time', n=1) / increments
    snow = snow_acc.diff(dim='time', n=1) / increments
    rain = np.insert(rain, 0, 0)
    snow = np.insert(snow, 0, 0)

    weather_icons = get_weather_icons(dset_city['WW'],time) 

    ax1 = plt.subplot(gs[i])

    ax1.set_xlim(time[0],time[-1])
    ts = ax1.plot(time, dset_city['2t'], label='$T$', color='darkcyan')
    ax1.set_ylabel('2 m $T$')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.grid(True, alpha=0.5)

    for dt, weather_icon, dewp in zip(time, weather_icons, dset_city['2t']):
        imagebox = OffsetImage(weather_icon, zoom=.025)
        ab = AnnotationBbox(imagebox, (mdates.date2num(dt), dewp), frameon=False)
        ax1.add_artist(ab)

    ax2 = ax1.twinx()
    ax2.set_xlim(time[0], time[-1])
    ts2 = ax2.plot(time, rain, label='Rain', color='dodgerblue')
    ts3 = ax2.plot(time, snow, label='Snow', color='orchid')
    ax2.set_ylim(bottom=0.1)
    ax2.set_ylabel('Precip. [mm h$^{-1}$]')
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.xaxis.set_major_formatter(DateFormatter('%d %b %HZ'))

    an = annotation(ax2, '%s (%3.1fN, %3.1fE, %d m)' % (city, dset_city.lat, dset_city.lon, height.values) ,
                        loc='upper left')

    if (len(cities) > 1) and (i+1 != len(cities)) : # then turn off the ticks 
        ax1.axes.get_xaxis().set_ticklabels([])
    else:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment('right')

    if i == 0: # then plot the legend 
        handles,labels = [],[]
        for ax in (ax1, ax2):
            for h,l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        plt.legend(handles,labels, fontsize=7)

    i += 1

fig.subplots_adjust(hspace=0.1)

# Build the name of the output image
filename = folder_images+'/meteogram_multicity.png' 

if debug:
    plt.show(block=True)
else:
    plt.savefig(filename, dpi=100, bbox_inches='tight')         
