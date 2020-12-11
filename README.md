# icon_forecasts
Download and plot ICON-EU data.

![Plotting sample](http://guidocioni.altervista.org/icon_forecasts/winds_jet/winds_jet_72.png)

In the following repository I include a fully-functional suite of scripts 
needed to download, merge and plot data from the ICON-EU model,
which is freely available at https://opendata.dwd.de/weather/.

The main script to be called (possibly through cronjob) is `copy_data.run`. 
There, the current run version is determined, and files are downloaded from the DWD server.
CDO is used to merge the files. At the end of the process one single NETCDF file with all the timesteps for every variable is created. We keep these files separated and merge them whe necessary in Python.

## Installation
This is not package! It is just a collection of scripts which can be run from `copy_data.run`. It was tested on Linux and MacOS; it will not run on Windows since it uses `bash`. To install it just clone the folder.

You need the following UNIX utilities to run the main script
- `GNU parallel` to parallelize the download and processing of data
- `ncftp` to upload pictures to FTP
- `cdo` for the preprocessing
- `wget` to download the files
- `bzip2` to decompress the downloaded files

The `python` installation can be re-created with the up-to-date `requirements.txt`. The script was succesfully tested on both `python 2.7.15` and `python 3.7.8`. The 2.7 version for now is the most stable.

The most important packages to have installed are 

- `numpy`
- `pandas`
- `metpy`
- `xarray`
- `dask`
- `basemap`
- `matplotlib`
- `seaborn`
- `scipy`
- `geopy`

## Running 

### Determining the run
The main script to be called, possibly through `crontab`, is `copy_data.run`. At the beginning of the script we check what is the most recent run available on server (through `get_last_run.py`) and compare it to the latest run that we processed in `MODEL_DATA_FOLDER` through a semaphore file `last_processed_run.txt`. If there is no file or the new run on server is more recent than this one we start the processing, otherwise we exit. This way we can easily set just one cron job every 2 hours and this will automatically take care of processing the right run. 
An example of a `cronjob` that you can use is 

```bash
50   */2      *     *     * /path/to/icon_forecasts/copy_data.run > /tmp/icon-eu/`/bin/date +\%Y\%m\%d\%H\%M\%S`-cron.log 2>&1
```
We use the `SHELL` variable to make sure the job is started with `bash` and the `BASH_ENV` to load some of the binaries that we need in the job.
he `.cron_jobs_default_load` looks like this on ubuntu

```bash
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/user/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/user/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/user/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/user/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# include all env vars that we need in our job
# export ....

```

### Inputs to be defined 
Most of the inputs needed to run the code are contained at the beginning of the main bash script `copy_data.run`. In particular `MODEL_DATA_FOLDER` where the processing is done (downloading of files and creation of pictures). 
`NCFTP_BOOKMARK` is the FTP bookmark to be defined in `ncftp` so that user and password don't need to be entered every time.
We use a conda environment with all the packages needed to run the download/processing of the data so that we can easily run into `crontab` without the need to load any additional packages.

### Parallelized donwload of data 
Downloading and merging the data is one of the process that can take more time depending on the connection.
For this reason this is fully parallelized making use of the GNU `parallel` utility.
```bash
#2-D variables
variables=("T_2M" "TD_2M" "U_10M")
${parallel} -j ${N_CONCUR_PROCESSES} download_merge_2d_variable_icon_eu ::: "${variables[@]}"

#3-D variables on pressure levels
variables=("T" "FI" "RELHUM" "U" "V")
${parallel} -j ${N_CONCUR_PROCESSES} download_merge_3d_variable_icon_eu ::: "${variables[@]}"
```
The list of variables to download using such parallelization is provided as bash array. 2-D and 3-D variables have different
routines: these are all defined in the common library `functions_download_dwd.sh`. The link to the DWD opendata server is also defined in this file.

### Parallelized plotting
Plotting of the data is done using Python, but anyone could potentially use other software. This is also parallelized
given that plotting routines are the most expensive part of the whole script and can take a lot of time (up to 2 hours
depending on the load).
This is make especially easier by the
fact that the plotting scripts can be given as argument the projection so we can parallelize across multiple projections
and script files, for example:
```bash
scripts=("plot_cape.py" "plot_convergence.py")

projections=("euratl" "it" "de")

${parallel} -j ${N_CONCUR_PROCESSES} ${python} ::: "${scripts[@]}" ::: "${projections[@]}"
```
Furthermore in every individual `python` script a parallelization using `multiprocessing.Pool` over chunks of the input timesteps is performed. This means that, using the same `${N_CONCUR_PROCESSES}`, different plotting istances will act over chunks of 10 timesteps each to speed up the processes. The chunk size can be changed in `utils.py`.
**NOTE**
Depending on what is passed to `multiprocessing.Pool.map` in `args` you could get an error since some objects cannot be pickled. Make sure that you're passing only the necessary arrays for the plotting and not additional objects (e.g. `pint` arrays created by `metpy` may be the culprit of the error).

Note that every Python script used for plotting has an option `debug=True` to allow some testing of the script before pushing it to production. When this option is activated the `PNG` figures will not be produced and the script will not be parallelized. Instead just 1 timestep will be processed and the figure will be shown in a window using the matplotlib backend.

### Upload of the pictures
PNG pictures are uploaded to a FTP server defined in `ncftp` bookmarks. This operation is parallelized with only 2 jobs because many FTP servers have limits on concurrent incoming connections.

### Additional files
ICON-EU invariant data are automatically download by `download_invariant_icon_eu`. Shapefiles are included in the repository but can be replaced. The file `soil_saturation.nc` is used to produce the map of soil moisture saturation and can be produced (if there is any need) with the Jupyter notebook, but in the future it will probably be created automatically.
