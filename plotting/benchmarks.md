#Code benchmarks

###plot_cape.py
Running with `chunks_size` = 10, `N_CONCUR_PROCESSES` = 4 and `it` projection - old default: script took 00:01:23
- Difference between `contourf` and `pcolormesh` is small for `it` projection, only 5 seconds. For `euratl` projection about 10 seconds. I think it's not worth it as the quality of the resulting plot is worse. 


###plot_tmax plot_tmin
These are the scripts that last longer...need to be optimized
