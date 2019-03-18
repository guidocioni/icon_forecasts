# icon_forecasts
Download and plot ICON-EU data.

In the following repository I include a fully-functional suite of scripts 
needed to download, merge and plot data from the ICON-EU model,
which is freely available at https://opendata.dwd.de/weather/.

The main script to be called (possibly through cronjob) is `copy_data.run`. 
There, the current run version is determined, and files are downloaded from the DWD server.
CDO is used to merge the files. At the end of the process one NETCDF file with all the variables and timesteps is created.

## Parallelized donwload of data 
Downloading and merging the data is one of the process that can take more time depending on the connection.
For this reason this is fully parallelized making use of the GNU `parallel` utility.
```bash
#2-D variables
variables=("T_2M" "TD_2M" "U_10M" "V_10M" "PMSL" "CAPE_ML" "VMAX_10M" "TOT_PREC" "CLCL" "CLCH" "CLCT" 
	   "SNOWLMT" "HZEROCL" "H_SNOW" "SNOW_GSP" "SNOW_CON" "RAIN_GSP" "RAIN_CON" "TMAX_2M" "TMIN_2M")
${parallel} -j 8 download_merge_2d_variable_icon_eu ::: "${variables[@]}"

#3-D variables on pressure levels
variables=("T" "FI" "RELHUM" "U" "V")
${parallel} -j 8 download_merge_3d_variable_icon_eu ::: "${variables[@]}"
```
The list of variables to download using such parallelization is provided as bash array. 2-D and 3-D variables have different
routines: these are all defined in the common library `functions_download_dwd.sh`. The link to the DWD opendata server is also defined in this file.

## Parallelized plotting
Plotting of the data is done using Python, but anyone could potentially use other software. This is also parallelized
given that plotting routines are the most expensive part of the whole script and can take a lot of time (up to 2 hours
depending on the load).
This is make especially easier by the
fact that the plotting scripts can be given as argument the projection so we can parallelize across multiple projections
and script files, for example:
```bash
scripts=("plot_cape.py" "plot_convergence.py" "plot_gph_t_500.py" "plot_gph_t_850.py" "plot_gph_thetae_850.py" 
"plot_hsnow.py" "plot_jetstream.py" "plot_pres_t2m_winds10m.py" "plot_rain_acc.py" "plot_rain_clouds.py" "plot_winds10m.py")

projections=("euratl" "it" "de")

${parallel} -j 8 ${python} ::: "${scripts[@]}" ::: "${projections[@]}"
```
Note that every Python script used for plotting has an option `debug=True` to allow some testing of the script before pushing it to production. When this option is activated the PNG figure will not be produced and the script will not be parallelized. Instead just 1 timestep will be processed and the figure will be shown in a window using the matplotlib backe
nd, thus easing the process of correcting details.

## Upload of the pictures
PNG pictures are uploaded to a FTP server defined in `ncftp` bookmarks.

## Additional files
ICON-EU invariant data are required by some NCL scripts. You can download them here: https://opendata.dwd.de/weather/icon/eu_nest/grib/00/invariant_data/
