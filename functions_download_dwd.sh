#Given a variable name and year-month-day-run as environmental variables download and merges the variable
################################################
parallelized_extraction(){
	# You need to pass a glob patter which will be tested for files
	# Wait at most 30 secs before the first grib2 appears
	i=0
	until [ `ls -1 ${1}.bz2 2>/dev/null | wc -l ` -gt 0 -o $i -ge 30 ]; do
		((i++))
	    sleep 1
	done
	# Then extract them, or if there is something just exit with an error
	while [ `ls -1 ${1}.bz2 2>/dev/null | wc -l ` -gt 0 ]; do
		ls ${1}.bz2| parallel -j+0 bzip2 -d '{}' 
	    sleep 1
	done
}
download_merge_2d_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -b -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	parallelized_extraction ${filename}
	cdo -f nc copy -mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.nc
	rm ${filename}
}
export -f download_merge_2d_variable_icon_eu
################################################
download_merge_3d_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.grib2"
	# 3 multiple connections so it should be faster
	filenames="icon-eu_europe_regular-lat-lon_pressure-level_${year}${month}${day}${run}_0[0-4][0-9]_*_${1}.grib2"
	wget -b -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filenames}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	filenames="icon-eu_europe_regular-lat-lon_pressure-level_${year}${month}${day}${run}_0[5-9][0-9]_*_${1}.grib2"
	wget -b -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filenames}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	filenames="icon-eu_europe_regular-lat-lon_pressure-level_${year}${month}${day}${run}_1[0-9][0-9]_*_${1}.grib2"
	wget -b -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filenames}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	#
	parallelized_extraction ${filename}
	cdo merge ${filename} ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}
	cdo -f nc copy ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_eur.nc
	rm ${1}_${year}${month}${day}${run}_eur.grib2
}
export -f download_merge_3d_variable_icon_eu
################################################
download_invariant_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_time-invariant_${year}${month}${day}${run}_HSURF.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/hsurf/"
	bzip2 -d ${filename}.bz2 
	cdo -f nc copy ${filename} HSURF_${year}${month}${day}${run}_eur.nc
	rm ${filename}
}
export -f download_invariant_icon_eu
################################################
download_merge_soil_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_soil-level_${year}${month}${day}${run}_*_3_${1}.grib2"
	wget -b -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	parallelized_extraction ${filename}
	cdo -f nc copy -mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.nc
	rm ${filename}
}
export -f download_merge_soil_variable_icon_eu