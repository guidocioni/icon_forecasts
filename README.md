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
For this reason this is fully parallelized making use of the Python `subprocess` package. From the main script `copy_data.run`
a Python script (`task_parallelism_general.py`), which takes care of the parallelization, is called. The total number of
concurrent processes can be defined inside this script. 

The list of variables to download using such parallelization is provided as bash array. 2-D and 3-D variables have different
routines: these are all defined in the common library `functions_download_dwd.sh` and are called using an environmental
variable. The link to the DWD opendata server is also defined in this file. 

## Parallelized plotting
Plotting of the data is done using NCL, but anyone could potentially use other software. This is also parallelized
given that plotting routines are the most expensive part of the whole script and can take a lot of time (up to 2 hours
depending on the load). Similary to what was done when downloading the files, plotting routines are parallelized using
`task_parallelism.py`.

## Upload of the pictures
PNG pictures are uploaded to a FTP server defined in `ncftp` bookmarks.

## Additional files
ICON-EU invariant data are required by some NCL scripts. You can download them here: https://opendata.dwd.de/weather/icon/eu_nest/grib/00/invariant_data/
