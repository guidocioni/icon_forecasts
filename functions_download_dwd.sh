#Given a variable name and year-month-day-run as environmental variables download and merges the variable
download_merge_2d_variable_icon_it_de()
{
	filename="ICON_EU_single_level_elements_${1}_${year}${month}${day}${run}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=4 -A "${filename}*" "https://opendata.dwd.de/weather/icon/eu_nest/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} mergetime ${filename}* ${1}_${year}${month}${day}${run}_eur.grib2
	${cdo} sellonlatbox,${min_lon_it},${max_lon_it},${min_lat_it},${max_lat_it} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_it.grib2
	${cdo} sellonlatbox,${min_lon_de},${max_lon_de},${min_lat_de},${max_lat_de} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_de.grib2
	rm ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}*
}
export -f download_merge_2d_variable_icon_it_de
################################################
download_merge_3d_variable_icon_it_de()
{
	filename="ICON_EU_pressure_level_elements_${1}_${year}${month}${day}${run}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=4 -A "${filename}*" "https://opendata.dwd.de/weather/icon/eu_nest/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} mergetime ${filename}* ${1}_${year}${month}${day}${run}_eur.grib2
	${cdo} sellonlatbox,${min_lon_it},${max_lon_it},${min_lat_it},${max_lat_it} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_it.grib2
	${cdo} sellonlatbox,${min_lon_de},${max_lon_de},${min_lat_de},${max_lat_de} ${1}_${year}${month}${day}${run}_eur.grib2 ${1}_${year}${month}${day}${run}_de.grib2
	rm ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}*
}
export -f download_merge_3d_variable_icon_it_de
################################################
download_merge_2d_variable_icon_eu()
{
	filename="ICON_EU_single_level_elements_${1}_${year}${month}${day}${run}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=4 -A "${filename}*" "https://opendata.dwd.de/weather/icon/eu_nest/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} mergetime ${filename}* ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}*
}
export -f download_merge_2d_variable_icon_eu
################################################
download_merge_3d_variable_icon_eu()
{
	filename="ICON_EU_pressure_level_elements_${1}_${year}${month}${day}${run}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=4 -A "${filename}*" "https://opendata.dwd.de/weather/icon/eu_nest/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} mergetime ${filename}* ${1}_${year}${month}${day}${run}_eur.grib2
	rm ${filename}*
}
export -f download_merge_3d_variable_icon_eu
################################################
download_merge_2d_variable_icon_globe()
{
	filename="ICON_iko_single_level_elements_world_${1}_${year}${month}${day}${run}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=4 -A "${filename}*" "https://opendata.dwd.de/weather/icon/global/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} -f nc copy -mergetime ${filename}* ${filename}.nc
	rm ${filename}*.grib2
}
export -f download_merge_2d_variable_icon_globe
################################################
download_merge_3d_variable_icon_globe()
{
	filename="ICON_iko_pressure_level_elements_world_${1}_${year}${month}${day}${run}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=4 -A "${filename}*" "https://opendata.dwd.de/weather/icon/global/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} -f nc copy -mergetime ${filename}* ${filename}.nc
	rm ${filename}*.grib2
}
export -f download_merge_3d_variable_icon_globe
################################################
download_merge_3d_variable_cosmo_d2()
{
	filename="cosmo-d2_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}"
	wget -r -nH -np -nv -nd --reject "index.html*" --cut-dirs=3 -A "${filename}*" "https://opendata.dwd.de/weather/nwp/cosmo-d2/grib/${run}"
	bzip2 -d ${filename}*
	for level in "300" "500" "700" "850" "950" "975" "1000" ; do
	 ${cdo} -f nc copy -setgridtype,curvilinear -mergetime cosmo-d2_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${level}_${1}.grib2 cosmo-d2_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_${level}_${1}.nc
	 rm cosmo-d2_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${level}_${1}.grib2
	done
	# First sort files numerically otherwise the levels get screwed up
	files_to_merge=`ls -v cosmo-d2_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.nc` 
	${cdo} merge ${files_to_merge} cosmo-d2_germany_pressure-level_${year}${month}${day}${run}_${1}.nc
	rm cosmo-d2_germany_rotated-lat-lon_pressure-level_${year}${month}${day}${run}_*_${1}.nc
}
export -f download_merge_3d_variable_cosmo_d2
################################################
download_merge_2d_variable_cosmo_d2()
{
	filename="cosmo-d2_germany_rotated-lat-lon_single-level_${year}${month}${day}${run}_*_${1}"
	wget -r -nH -np -nv -nd --reject "index.html" --cut-dirs=3 -A "${filename}*" "https://opendata.dwd.de/weather/nwp/cosmo-d2/grib/${run}"
	bzip2 -d ${filename}* 
	${cdo} -f nc copy  -setgridtype,curvilinear -mergetime ${filename}* cosmo-d2_single-level_${1}_${year}${month}${day}${run}.nc
	rm ${filename}*
}
export -f download_merge_2d_variable_cosmo_d2


