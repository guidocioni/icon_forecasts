load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file
wktype="png"
wktype@wkWidth = 1100
wktype@wkHeight = 1100

level=0

fils = systemfunc ("ls /scratch/local1/m300382/icon_forecasts/ICON_*.nc") ; file paths
f_rain    = addfiles (fils, "r")  
ListSetType (f_rain, "cat")

invariant = addfile("~/icon_forecasts/ICON_EU_invariant.nc", "r")
lsm = invariant->lsm(0,:,:)
soil_type= invariant->SOILTYP(0,:,:)

;  soil type               3       4      5      6     7       8
soil_type_saturation = (/0.364, 0.445, 0.455, 0.475, 0.507, 0.863/)

soil_saturation_3D = new(dimsizes(soil_type), double)
copy_VarMeta(soil_type, soil_saturation_3D)

do i=0, dimsizes(soil_type_saturation)-1
  x1_1D      = ndtooned(soil_saturation_3D)                  ; convert to 1D array
  x2_1D      = ndtooned(soil_type)
  indices  = ind(x2_1D.eq.i+3)             
  x1_1D(indices) = soil_type_saturation(i)                           ; set all  array syntax
  soil_saturation_3D = onedtond(x1_1D, dimsizes(soil_saturation_3D)); return to the original array
 delete([/x1_1D, x2_1D, indices/])
end do 

plot_description="Soil Moisture (in percentage of the saturation value)"

 w_so = f_rain[:]->W_SO

 rho_w=1000.0
 depth=tofloat(f_rain[:]->depth)
 depth(0)=depth(0)*2
 do k=0,dimsizes(depth)-1
     ;normalize over depth
     w_so(:,k,:,:)=w_so(:,k,:,:)/(depth(k)*rho_w)
 end do

w_so=mask(w_so, conform(w_so, lsm, (/2, 3/) ).lt.0.5, False)
w_so_percentage=w_so/conform(w_so, soil_saturation_3D, (/2, 3/) )
copy_VarMeta(w_so, w_so_percentage)
w_so_percentage=w_so_percentage*100.


 lon = f_rain[:]->lon 
 lat = f_rain[:]->lat
 time= f_rain[:]->time 

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time

 delete([/fils, f_rain/])

; create plot

 ResC = True
 ResC@gsnDraw = False
 ResC@gsnFrame = False
 ResC@gsnAddCyclic = False  
 ResC@gsnMaximize   = True         ; Maximize plot in frame.
 ResC@gsnLeftString = ""
 ResC@gsnRightString = ""
 ResC@gsnStringFontHeightF=0.01
 ResC@cnFillMode = "RasterFill"

 ResC2=ResC

 ResC@cnFillOn = True ; do color fill
 ResC@cnLinesOn=False
 ResC@mpFillOn     = True 
 ResC@mpOceanFillColor = "skyblue"
 ResC@mpInlandWaterFillColor = "skyblue"
 ResC@mpLimitMode = "LatLon"                   ;
 ResC@mpMaxLonF    = max(lon)                      ; specify the plot domain
 ResC@mpMinLonF    = min(lon)                      ;
 ResC@mpMinLatF    = min(lat)                     ;
 ResC@mpMaxLatF    = max(lat)                    ;
 ResC@mpDataBaseVersion  = "MediumRes"
 ResC@mpGeophysicalLineThicknessF = 2
 ResC@mpGeophysicalLineColor ="black"
 ResC@mpOutlineBoundarySets = "AllBoundaries"     ; more outlines
 ResC@mpNationalLineThicknessF = 2
 ResC@mpDataSetName         = "Earth..4"
 ; ResC@mpOutlineOn = True
 ResC@tmXBLabelFontHeightF = 0.01
 ResC@pmTickMarkDisplayMode = True

ResC@cnLevelSelectionMode = "ExplicitLevels"
ResC@cnLevels = ispan(0, 100, 5)

ResC@cnFillPalette = "GMT_drywet"

ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.04
ResC@pmLabelBarOrthogonalPosF = 0.
ResC@lbLabelFontHeightF = 0.005

ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

txres                       = True
txres@txPerimOn             = True
txres@txBackgroundFillColor = "White"
txres@txFontHeightF         = 0.011

amres=True 
amres@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres@amOrthogonalPosF = 0.5    ; This is the bottom edge of the plot.
amres@amJust = "BottomLeft"

amres2=True 
amres2@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres2@amOrthogonalPosF = -0.5    ; This is the bottom edge of the plot.
amres2@amJust = "TopLeft"

amres3=True 
amres3@amParallelPosF   = 0.5    ; This is the right edge of the plot.
amres3@amOrthogonalPosF = -0.5    ; This is the bottom edge of the plot.
amres3@amJust = "TopRight"

 do time_i=0,dimsizes(time)-1
 ResC@gsnCenterString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
 wks = gsn_open_wks(wktype,"soil_moisture_"+date_string(time_i))
 
  plot=gsn_csm_contour_map(wks, w_so_percentage(time_i,level,:,:), ResC)

   text=gsn_create_text(wks, plot_description, txres)
   description=gsn_add_annotation(plot, text, amres)
   text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

 draw(plot)
 frame(wks)
end do



end
