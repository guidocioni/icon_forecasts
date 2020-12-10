# crontab settings
This document contains some info on how to set your cron jobs properly.

We only use 0, 6, 12, 18 runs. The files for these runs are usually *all* ready on `opendata.dwd.de` at
- 04:15 GMT for 00Z run
- 11:30 GMT for 06Z run
- 16:15 GMT for 12Z run
- 22:15 GMT for 18Z run

This means that we have to use different timing in `cron` as the timezone cannot be yet specified.
With DST (daylight saving time) between 28 March and 31 October usually, the difference is only one hour, so

- 05:15 CEST for 00Z run
- 12:30 CEST for 06Z run
- 17:15 CEST for 12Z run
- 23:15 CEST for 18Z run

Without DST we have to add another hour as the difference is of 2 hours in CET.

- 06:15 CET for 00Z run
- 13:30 CET for 06Z run
- 18:15 CET for 12Z run
- 00:15 CET for 18Z run

Here is an example of cron jobs configuration. We use the `SHELL` variable to make sure the job is started with `bash` and the `BASH_ENV` to load some of the binaries that we need in the job.

```bash
SHELL=/bin/bash
BASH_ENV=/home/user/.cron_jobs_default_load
# icon-eu forecasts
15   6      *     *     * /home/user/icon_forecasts/copy_data.run > /tmp/icon-eu/`/bin/date +\%Y\%m\%d\%H\%M\%S`-cron.log 2>&1
30   13      *     *     * /home/user/icon_forecasts/copy_data.run > /tmp/icon-eu/`/bin/date +\%Y\%m\%d\%H\%M\%S`-cron.log 2>&1
15   18      *     *     * /home/user/icon_forecasts/copy_data.run > /tmp/icon-eu/`/bin/date +\%Y\%m\%d\%H\%M\%S`-cron.log 2>&1
59   23      *     *     * /home/user/icon_forecasts/copy_data.run > /tmp/icon-eu/`/bin/date +\%Y\%m\%d\%H\%M\%S`-cron.log 2>&1
```
we save the output of the scripts always in the `/tmp/icon-eu/` folder with a name created using the current date.
The `.cron_jobs_default_load` looks like this on ubuntu
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
