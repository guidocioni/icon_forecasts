from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run', help='Run to search for, otherwise defaults to all runs available for the model',
                    required=False, default=None)
parser.add_argument('-v2d', '--vars_2d', help='List of 2d variables to be checked',
                    required=False, default=None, nargs='+')
parser.add_argument('-v3d', '--vars_3d', help='List of 3d variables to be checked',
                    required=False, default=['t'], nargs='+')
parser.add_argument('-l', '--levels_3d', help='List of 3d levels to be checked',
                    required=False, default=['850'], nargs='+')

args = parser.parse_args()

var_2d_list = ['alb_rad', 'alhfl_s', 'ashfl_s', 'asob_s', 'asob_t', 'aswdifd_s', 'aswdifu_s',
               'aswdir_s', 'athb_s', 'cape_con', 'cape_ml', 'clch', 'clcl', 'clcm', 'clct',
               'clct_mod', 'cldepth', 'h_snow', 'hbas_con', 'htop_con', 'htop_dc', 'hzerocl',
               'pmsl', 'ps', 'qv_2m', 'qv_s', 'rain_con', 'rain_gsp', 'relhum_2m', 'rho_snow',
               'runoff_g', 'runoff_s', 'snow_con', 'snow_gsp', 'snowlmt', 'synmsg_bt_cl_ir10.8',
               't_2m', 't_g', 't_snow', 'tch', 'tcm', 'td_2m', 'tmax_2m', 'tmin_2m', 'tot_prec',
               'u_10m', 'v_10m', 'vmax_10m', 'w_snow', 'w_so', 'ww', 'z0']

var_3d_list = ['clc', 'fi', 'omega', 'p',
               'qv', 'relhum', 't', 'tke', 'u', 'v', 'w']


def get_url_paths(url, ext='', prefix='', params={}):
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if (
        node.get('href').endswith(ext)) & (node.get('href').startswith(prefix))]
    return parent


def find_file_name(vars_2d=None,
                   vars_3d=None,
                   levels_3d=None,
                   base_url="https://opendata.dwd.de/weather/nwp",
                   model_url="icon-eu/grib",
                   date_string=None,
                   run_string=None):
    if run_string in ['00', '06', '12', '18']:
        f_times = list(range(0, 79)) + list(range(81, 121, 3))
    else:
        f_times = list(range(0, 31))
    #
    if type(f_times) is not list:
        f_times = [f_times]
    if (vars_2d is None) and (vars_3d is None):
        raise ValueError(
            'You need to specify at least one 2D or one 3D variable')

    if vars_2d is not None:
        if type(vars_2d) is not list:
            vars_2d = [vars_2d]
    if vars_3d is not None:
        if levels_3d is None:
            raise ValueError(
                'When specifying 3d coordinates you also need levels')
        if type(vars_3d) is not list:
            vars_3d = [vars_3d]
    if levels_3d is not None:
        if type(levels_3d) is not list:
            levels_3d = [levels_3d]

    data = {'run': [], 'variable': [], 'status': [],
            'avail_tsteps': [], 'missing_tsteps': []}
    if vars_2d is not None:
        for var in vars_2d:
            if var not in var_2d_list:
                raise ValueError('accepted 2d variables are %s' % var_2d_list)
            data['variable'].append(var)
            data['run'].append('%s%s' % (date_string, run_string))
            urls_to_check = []
            for f_time in f_times:
                var_url = "icon-eu_europe_regular-lat-lon_single-level"
                urls_to_check.append("%s/%s/%s/%s/%s_%s%s_%03d_%s.grib2.bz2" %
                                     (base_url, model_url, run_string, var,
                                      var_url, date_string, run_string, f_time, var.upper()))
            urls_on_server = get_url_paths("%s/%s/%s/%s/" % (base_url, model_url, run_string, var),
                                           'grib2.bz2', prefix=var_url)
            if set(urls_to_check).issubset(urls_on_server):
                data['status'].append('all files available')
                data['avail_tsteps'].append(len(urls_to_check))
                data['missing_tsteps'].append(0)
            else:
                intersection = set(urls_to_check).intersection(
                    set(urls_on_server))
                data['status'].append('incomplete')
                data['avail_tsteps'].append(len(intersection))
                data['missing_tsteps'].append(len(f_times) - len(intersection))

    if vars_3d is not None:
        for var in vars_3d:
            if var not in var_3d_list:
                raise ValueError('accepted 3d variables are %s' % var_3d_list)
            data['variable'].append(var)
            data['run'].append('%s%s' % (date_string, run_string))
            urls_to_check = []
            for plev in levels_3d:
                for f_time in f_times:
                    var_url = "icon-eu_europe_regular-lat-lon_pressure-level"
                    urls_to_check.append("%s/%s/%s/%s/%s_%s%s_%03d_%s_%s.grib2.bz2" %
                                         (base_url, model_url, run_string, var,
                                          var_url, date_string, run_string, f_time, plev, var.upper()))
            urls_on_server = get_url_paths("%s/%s/%s/%s/" % (base_url, model_url, run_string, var),
                                           'grib2.bz2', prefix=var_url)
            if set(urls_to_check).issubset(urls_on_server):
                data['status'].append('all files available')
                data['avail_tsteps'].append(len(urls_to_check))
                data['missing_tsteps'].append(0)
            else:
                intersection = set(urls_to_check).intersection(
                    set(urls_on_server))
                data['status'].append('incomplete')
                data['avail_tsteps'].append(len(intersection))
                data['missing_tsteps'].append(len(f_times) - len(intersection))

    df = pd.DataFrame(data)

    return df


def get_most_recent_run(run=None, vars_2d=None, vars_3d=['t'],
                        levels_3d=['850']):
    today_string = datetime.now().strftime('%Y%m%d')
    yesterday_string = (datetime.today() -
                        timedelta(days=1)).strftime('%Y%m%d')
    if run is None:
        runs = ['00', '03', '06', '09', '12', '15', '18', '21']
    else:
        runs = [run]
    temp = []
    for date_string in [yesterday_string, today_string]:
        for run_string in runs:
            temp.append(find_file_name(vars_2d=vars_2d,
                                       vars_3d=vars_3d,
                                       levels_3d=levels_3d,
                                       date_string=date_string,
                                       run_string=run_string))
    final = pd.concat(temp)
    sel_run = final.loc[final.status == 'all files available', 'run'].max()
    return final, sel_run


if __name__ == "__main__":
    final, sel_run = get_most_recent_run(run=args.run, vars_2d=args.vars_2d,
                        vars_3d=args.vars_3d, levels_3d=args.levels_3d)
    print(final)
