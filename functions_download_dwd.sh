#Given a variable name and year-month-day-run as environmental variables download and merges the variable
################################################
download_merge_2d_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
	cdo -f nc copy -mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.nc
	rm ${filename}
}
export -f download_merge_2d_variable_icon_eu
################################################
download_merge_3d_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2 
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
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
	cdo -f nc copy -mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.nc
	rm ${filename}
}
export -f download_merge_soil_variable_icon_eu