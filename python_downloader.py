import os
import xarray as xr
from datetime import datetime
import requests
import bz2
from multiprocessing import Pool, cpu_count
from glob import glob


var_2d_list = ['alb_rad','alhfl_s','ashfl_s','asob_s','asob_t','aswdifd_s','aswdifu_s',
          'aswdir_s','athb_s','cape_con','cape_ml','clch','clcl','clcm','clct',
          'clct_mod','cldepth','h_snow','hbas_con','htop_con','htop_dc','hzerocl',
          'pmsl','ps','qv_2m','qv_s','rain_con','rain_gsp','relhum_2m','rho_snow',
          'runoff_g','runoff_s','snow_con','snow_gsp','snowlmt','synmsg_bt_cl_ir10.8',
          't_2m','t_g','t_snow','tch','tcm','td_2m','tmax_2m','tmin_2m','tot_prec',
          'u_10m','v_10m','vmax_10m','w_snow','w_so','ww','z0']

var_3d_list = ['clc','fi','omega','p','qv','relhum','t','tke','u','v','w']

pressure_levels = [1000, 950, 925, 900, 875, 850, 825,
                   800, 775, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50]


def get_run():
    now = datetime.now()
    date_string = now.strftime('%Y%m%d')
    utc_now = datetime.utcnow()

    if (utc_now.replace(hour=4, minute=0, second=0, microsecond=0) 
        <= utc_now < utc_now.replace(hour=9, minute=0, second=0, microsecond=0)):
        run="00"
    elif (utc_now.replace(hour=9, minute=0, second=0, microsecond=0) 
        <= utc_now < utc_now.replace(hour=16, minute=0, second=0, microsecond=0)):
        run="06"
    elif (utc_now.replace(hour=16, minute=0, second=0, microsecond=0) 
        <= utc_now < utc_now.replace(hour=21, minute=0, second=0, microsecond=0)):
        run="12"
    elif (utc_now.replace(hour=21, minute=0, second=0, microsecond=0) 
        <= utc_now):
        run="18"

    return now.strftime('%Y%m%d')+run, run


def find_file_name(vars_2d=None,
                   vars_3d=None,
                   f_times=0, 
                   base_url = "https://opendata.dwd.de/weather/nwp",
                   model_url = "icon-eu/grib"):
    '''Find file names to be downloaded given input variables and
    a forecast lead time f_time (in hours).
    - vars_2d, a list of 2d variables to download, e.g. ['t_2m']
    - vars_3d, a list of 3d variables to download with pressure
      level, e.g. ['t@850','fi@500']
    - f_times, forecast steps, e.g. 0 or list(np.arange(1, 79))
    Note that this function WILL NOT check if the files exist on
    the server to avoid wasting time. When they're passed
    to the download_extract_files function if the file does not
    exist it will simply not be downloaded.
      '''
    date_string, run_string = get_run()
    if type(f_times) is not list:
        f_times = [f_times]
    if (vars_2d is None) and (vars_3d is None):
        raise ValueError('You need to specify at least one 2D or one 3D variable')

    if vars_2d is not None:
        if type(vars_2d) is not list:
            vars_2d = [vars_2d]
    if vars_3d is not None:
        if type(vars_3d) is not list:
            vars_3d = [vars_3d]

    urls = []
    for f_time in f_times:
        if vars_2d is not None:
            for var in vars_2d:
                if var not in var_2d_list:
                    raise ValueError('accepted 2d variables are %s' % var_2d_list)
                var_url="icon-eu_europe_regular-lat-lon_single-level"
                urls.append("%s/%s/%s/%s/%s_%s_%03d_%s.grib2.bz2" % 
                            (base_url, model_url, run_string, var,
                              var_url, date_string, f_time, var.upper()) )
        if vars_3d is not None:
            for var in vars_3d:
                var_t, plev = var.split('@')
                if var_t not in var_3d_list:
                    raise ValueError('accepted 3d variables are %s' % var_3d_list)
                var_url="icon-eu_europe_regular-lat-lon_pressure-level"
                urls.append("%s/%s/%s/%s/%s_%s_%03d_%s_%s.grib2.bz2" % 
                            (base_url, model_url, run_string, var_t,
                              var_url, date_string, f_time, plev, var_t.upper()) )

    return urls


def download_extract_files(urls):
    '''Given a list of urls download and bunzip2 them.
    Return a list of the path of the extracted files'''

    if type(urls) is list:
        urls_list = urls
    else:
        urls_list = [urls]

    # We only parallelize if we have a number of files
    # larger than the cpu count 
    if len(urls_list) > cpu_count():    
        pool = Pool(cpu_count())
        results = pool.map(download_extract_url, urls_list)
        pool.close()
        pool.join()
    else:
        results = []
        for url in urls_list:
            results.append(download_extract_url(url))

    return results


def download_extract_url(url, folder='/tmp/icon-eu/python_test/'):
    filename = folder+os.path.basename(url).replace('.bz2','')

    if os.path.exists(filename):
        extracted_files = filename
    else:
        r = requests.get(url, stream=True)
        if r.status_code == requests.codes.ok:
            with r.raw as source, open(filename, 'wb') as dest:
                dest.write(bz2.decompress(source.read()))
            extracted_files = filename
        else:
            return None

    return extracted_files


def get_dset(vars_2d=[], vars_3d=[], f_times=0):
    if vars_2d or vars_3d:
        date_string, _ = get_run()
        urls = find_file_name(vars_2d=vars_2d,
                              vars_3d=vars_3d,
                              f_times=f_times)
        fils = download_extract_files(urls)

    return fils


# def merge_files(vars_2d=[], vars_3d=[], folder='/tmp/'):
#     date_string, _ = get_run()
#     if vars_2d is not None:
#         if type(vars_2d) is not list:
#             vars_2d = [vars_2d]
#     if vars_3d is not None:
#         if type(vars_3d) is not list:
#             vars_3d = [vars_3d]

#     merged_files = []

#     for var in vars_2d:
#         merged_file = folder + var.upper() + '_' + date_string + '_eur.grib2'
#         files_to_merge = glob(folder+'icon-eu_europe_regular-lat-lon_single-level_'+date_string+'_*_'+var.upper()+'.grib2')
#         if len(files_to_merge) > 0:
#             if os.path.exists(merged_file) == False:
#                 os.system('cat %s > %s' % (' '.join(files_to_merge), merged_file))
#                 #
#                 for fname in files_to_merge:
#                     if os.path.isfile(fname):
#                         os.remove(fname)

#     for var in vars_3d:
#         var_t, plev = var.split('@')
#         merged_file = folder + var_t.upper() + '_' + date_string + '_eur.grib2'
#         files_to_merge = glob(folder+'icon-eu_europe_regular-lat-lon_pressure-level_'+date_string+'_*_'+var_t.upper()+'.grib2')
#         if len(files_to_merge) > 0:
#             if os.path.exists(merged_file) == False:
#                 os.system('cat %s > %s' % (' '.join(files_to_merge), merged_file))
#                 #
#                 for fname in files_to_merge:
#                     if os.path.isfile(fname):
#                         os.remove(fname)


f_steps = list(range(0, 79)) + list(range(81, 121, 3))
vars_3d_download = ['clc','fi','relhum','t','u','v','w']
pressure_levels_download = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150]
vars_3d = [v+'@'+str(p) for v in vars_3d_download for p in pressure_levels_download]
vars_2d = ['cape_ml','clch','clcl','clct', 'h_snow','hzerocl',
          'pmsl','rain_con','rain_gsp','snow_con','snow_gsp','snowlmt',
          't_2m','td_2m','tmax_2m','tmin_2m','tot_prec',
          'u_10m','v_10m','vmax_10m','w_snow','ww']

# get_dset(vars_2d=vars_2d,
#          vars_3d=vars_3d,
#          f_times=f_steps)

get_dset(vars_2d=vars_2d, f_times=f_steps)
