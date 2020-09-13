#Given a variable name and year-month-day-run as environmental variables download and merges the variable
download_merge_2d_variable_icon_it_de()
{
	filename="icon-eu_europe_regular-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	bzip2 -d ${filename}.bz2 
	${cdo} mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.grib2
	${cdo} sellonlatbox,${min_lon_it},${max_lon_it},${min_lat_it},${max_lat_it} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_it.grib2
	${cdo} sellonlatbox,${min_lon_de},${max_lon_de},${min_lat_de},${max_lat_de} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_de.grib2
	rm ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}
}
export -f download_merge_2d_variable_icon_it_de
################################################
download_merge_3d_variable_icon_it_de()
{
	filename="icon-eu_europe_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2 
	${cdo} mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.grib2
	${cdo} sellonlatbox,${min_lon_it},${max_lon_it},${min_lat_it},${max_lat_it} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_it.grib2
	${cdo} sellonlatbox,${min_lon_de},${max_lon_de},${min_lat_de},${max_lat_de} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_de.grib2
	rm ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}
}
export -f download_merge_3d_variable_icon_it_de
################################################
download_merge_2d_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
	${cdo} mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.grib2
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
	${cdo} mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}
}
export -f download_merge_3d_variable_icon_eu
################################################
download_merge_soil_variable_icon_eu()
{
	filename="icon-eu_europe_regular-lat-lon_soil-level_${year}${month}${day}${run}_*_0_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
	${cdo} mergetime ${filename} ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}
}
export -f download_merge_soil_variable_icon_eu
################################################
download_merge_2d_variable_icon_globe()
{
	filename="icon_global_icosahedral_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2 
	${cdo} -f nc copy -mergetime ${filename} icon_global_icosahedral_single-level_${1}_${year}${month}${day}${run}.nc
	rm ${filename}
}
export -f download_merge_2d_variable_icon_globe
################################################
download_merge_3d_variable_icon_globe()
{
	filename="icon_global_icosahedral_pressure-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
	for level in "300" "500" "700" "850" "950" "1000" ; do
	 ${cdo} -f nc copy -mergetime icon_global_icosahedral_pressure-level_${year}${month}${day}${run}_*_${level}_${1}.grib2 icon_global_icosahedral_pressure-level_${year}${month}${day}${run}_${level}_${1}.nc
	 rm icon_global_icosahedral_pressure-level_${year}${month}${day}${run}_*_${level}_${1}.grib2
	done
	# First sort files numerically otherwise the levels get screwed up
	files_to_merge=`ls -v icon_global_icosahedral_pressure-level_${year}${month}${day}${run}_*_${1}.nc` 
	${cdo} merge ${files_to_merge} icon_global_icosahedral_pressure-level_${1}_${year}${month}${day}${run}.nc
	rm icon_global_icosahedral_pressure-level_${year}${month}${day}${run}_*_${1}.nc
}
export -f download_merge_3d_variable_icon_globe
################################################
download_merge_3d_variable_cosmo_d2()
{
	filename="cosmo-d2_germany_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/cosmo-d2/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
	for level in "200" "300" "400" "500" "700" "850" "950" "975" "1000" ; do
	 ${cdo} -f nc copy -mergetime cosmo-d2_germany_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${level}_${1}.grib2 cosmo-d2_germany_regular-lat-lon_pressure-level_${year}${month}${day}${run}_${level}_${1}.nc
	 rm cosmo-d2_germany_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${level}_${1}.grib2
	done
	# First sort files numerically otherwise the levels get screwed up
	files_to_merge=`ls -v cosmo-d2_germany_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.nc` 
	${cdo} merge ${files_to_merge} cosmo-d2_germany_pressure-level_${year}${month}${day}${run}_${1}.nc
	rm cosmo-d2_germany_regular-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.nc
}
export -f download_merge_3d_variable_cosmo_d2
################################################
download_merge_2d_variable_cosmo_d2()
{
	filename="cosmo-d2_germany_regular-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/cosmo-d2/grib/${run}/${1,,}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2 
	${cdo} -f nc copy -mergetime ${filename} cosmo-d2_single-level_${1}_${year}${month}${day}${run}.nc
	rm ${filename}
}
export -f download_merge_2d_variable_cosmo_d2
################################################
download_merge_2d_variable_cosmo_d2_eps()
{
	filename="cosmo-d2-eps_germany_rotated-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/cosmo-d2-eps/grib/${run}/${1}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
    # Split the minutes
    for f in cosmo-d2-eps_germany_rotated-lat-lon_single-level_${year}${month}${day}${run}_*_${1}.grib2
    do 
        for minutes in 00 15 30 45
        do
            NEW=${f%_$1.grib2}_${minutes}_${1}.grib2
            ${cdo} select,minute=${minutes} ${f} ${NEW}
        done
    done
}
export -f download_merge_2d_variable_cosmo_d2_eps
################################################
download_merge_3d_variable_cosmo_d2_eps()
{
	filename="cosmo-d2-eps_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/cosmo-d2-eps/grib/${run}/${1}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2
    # Split the minutes   
    for f in cosmo-d2-eps_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.grib2
    do 
        for minutes in 00 15 30 45
        do
            NEW=${f%_$1.grib2}_${minutes}_${1}.grib2
            ${cdo} select,minute=${minutes} ${f} ${NEW}
        done
    done
}
export -f download_merge_3d_variable_cosmo_d2_eps
################################################
download_merge_2d_variable_icon_eps()
{
	filename="icon-eps_global_icosahedral_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eps/grib/${run}/${1}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2 
}
export -f download_merge_2d_variable_icon_eps
################################################
download_merge_2d_variable_icon_eu_eps()
{
	filename="icon-eu-eps_europe_icosahedral_single-level_${year}${month}${day}${run}_*_${1}.grib2"
	wget -r -nH -np -nv -nd --reject "index.html" --cut-dirs=3 -A "${filename}.bz2" "https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib/${run}/${1}/"
	echo 'Extracting files'
	bzip2 -d ${filename}.bz2 
}
export -f download_merge_2d_variable_icon_eu_eps
