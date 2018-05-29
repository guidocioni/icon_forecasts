load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls /scratch/local1/m300382/icon_forecasts/*.nc") ; file paths
f_rain    = addfiles (fils, "r")  
ListSetType (f_rain, "cat")

plot_description="High clouds (red shadings), low clouds (grey shadings), rain and snow"


 clcm=f_rain[:]->CLCL(:,0,:,:)
 clch=f_rain[:]->CLCH(:,0,:,:)
  mslp=f_rain[:]->prmsl

 ; depth_snow=f_rain[:]->sd 
 depth_snow=f_rain[:]->SNOW_CON + f_rain[:]->SNOW_GSP
 copy_VarMeta(f_rain[:]->SNOW_CON, depth_snow)
 ; tot_prec=f_rain[:]->tp

 tot_prec=f_rain[:]->RAIN_CON + f_rain[:]->RAIN_GSP
 copy_VarMeta(f_rain[:]->RAIN_CON, tot_prec)

 mslp = mslp/100.

 lon = f_rain[:]->lon 
 lat = f_rain[:]->lat
 time= f_rain[:]->time 

 date_string=time/60
 run_time=str_sub_str(time@units,"minutes since ","")
 run_string="Run: "+run_time

 delete([/fils, f_rain/])

 rain = tot_prec 
 rain = 0.0

 delete(rain@units)
 
 do i=1,dimsizes(time)-1
    rain(i,:,:)=tot_prec(i,:,:)-tot_prec(i-1,:,:) 
 end do 

; depth_snow=depth_snow
change_snow=depth_snow
change_snow=0

do i=1,dimsizes(time)-1
    change_snow(i,:,:)=depth_snow(i,:,:)-depth_snow(i-1,:,:)
end do 

;change_snow=mask(change_snow, change_snow .le. 0.5, False)

; create plot

 ResC = True
 ResC@gsnDraw = False
 ResC@gsnFrame = False
 ResC@gsnAddCyclic = False  
 ResC@gsnMaximize   = True         ; Maximize plot in frame.
 ResC@gsnLeftString = ""
 ResC@gsnRightString = ""
 ResC@cnFillMode = "RasterFill"

 ResC@cnFillOn = True ; do color fill
 ;ResC@cnFillMode = "rasterfill"
 ResC@cnLinesOn=False
 ResC2=ResC
 ResC3=ResC
 ResC4=ResC 
 ResC5=ResC

 ResC@mpFillOn     = False
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

ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = 10
ResC@cnMaxLevelValF = 110 
ResC@cnLevelSpacingF = 10
ResC@lbLabelBarOn = False 
ResC@cnInfoLabelOn = False 
ResC@cnLineLabelsOn = False 

colors=(/"white","grey80","grey75","grey70","grey65","grey60","grey55"/)
cmap_r = span_named_colors(colors,False)
; cmap_r(0,:)         = 0.0    ; Fully transparent
ResC@cnFillPalette = cmap_r

ResC2@cnLevelSelectionMode = "ExplicitLevels"
ResC2@cnLevels = (/0.5, 1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 5, 6, 7, 8, 9, 10, 15, 20, 25/)

cmap2_r              = read_colormap_file("precip2_17lev")
cmap2_r(0,:)=0.0
ResC2@cnFillPalette = cmap2_r
ResC2@cnFillMode = "RasterFill"

ResC2@lbOrientation = "Horizontal"
ResC2@pmLabelBarHeightF = 0.04
; ResC2@pmLabelBarWidthF = 0.15
ResC2@pmLabelBarOrthogonalPosF = 0.
ResC2@lbLabelFontHeightF = 0.005
ResC2@lbTitleOn = True 
ResC2@lbTitleString = "Rain [mm]"
ResC2@lbTitlePosition = "Right"
ResC2@lbTitleFontHeightF = 0.008
ResC2@lbTitleDirection = "Across"

ResC3@cnLevelSelectionMode = "ManualLevels"
ResC3@cnMinLevelValF = 30
ResC3@cnMaxLevelValF = 100 
ResC3@cnLevelSpacingF = 10
ResC3@cnFillOpacityF = 0.4
ResC3@lbLabelBarOn = False 
ResC3@cnInfoLabelOn = False 
ResC3@cnLineLabelsOn = False 

cmap3_r              = read_colormap_file("WhiteYellowOrangeRed")
cmap3_r(0,:)         = 0.0    ; Fully transparent
ResC3@cnFillPalette = cmap3_r

ResC4@cnFillOn = False ; do color fill
ResC4@cnLinesOn=True
ResC4@cnLevelSelectionMode = "ManualLevels"
ResC4@cnMinLevelValF = round(min(mslp), 0)
ResC4@cnMaxLevelValF = round(max(mslp), 0)
ResC4@cnInfoLabelOn = False  
ResC4@cnLevelSpacingF = 6
ResC4@cnLineThicknessF = 3
ResC4@cnLineColor = "grey40"
ResC4@cnLineLabelFontHeightF = 0.004

cmap5_r              = read_colormap_file("MPL_PuRd")
; cmap5_r(0,4)         = 0.2   
ResC5@cnFillPalette = cmap5_r(20:,:)
ResC5@pmLabelBarHeightF = 0.04
ResC5@lbLabelFontHeightF = 0.005
ResC5@pmLabelBarOrthogonalPosF = 0.05
ResC5@cnLevelSelectionMode = "ExplicitLevels"
ResC5@cnLevels = (/1, 1.5, 2, 2.5, 3, 4, 5, 10, 20/)
ResC5@lbTitleOn = True 
ResC5@lbTitleString = "Snow [cm]"
ResC5@lbTitlePosition = "Right"
ResC5@lbTitleFontHeightF = 0.008
ResC5@lbTitleDirection = "Across"

ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.02

txres                       = True
txres@txPerimOn             = True
txres@txBackgroundFillColor = "White"
txres@txFontHeightF         = 0.01

amres=True 
amres@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres@amOrthogonalPosF = 0.5    ; This is the bottom edge of the plot.
amres@amJust = "BottomLeft"


amres2=True 
amres2@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres2@amOrthogonalPosF = -0.5    ; This is the bottom edge of the plot.
amres2@amJust = "TopLeft"

 do time_i=1,dimsizes(time)-1
 ResC@gsnCenterString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
 wks = gsn_open_wks("png","precip_clouds_"+date_string(time_i))
 
 plot = gsn_csm_contour_map(wks,clcm(time_i,:,:),ResC)
 over_clouds=gsn_csm_contour(wks,clch(time_i,:,:), ResC3)
 over=gsn_csm_contour(wks, rain(time_i,:,:), ResC2)
 over_snow=gsn_csm_contour(wks, change_snow(time_i,:,:), ResC5)

 over_mslp=gsn_csm_contour(wks, mslp(time_i,:,:), ResC4)

 text=gsn_create_text(wks, plot_description, txres)
 description=gsn_add_annotation(plot, text, amres)

 text2=gsn_create_text(wks, run_string, txres)
 run_annotation=gsn_add_annotation(plot, text2, amres2)

 overlay(plot,over_clouds)
 overlay(plot,over)
 ;overlay(plot,over_snow)
 overlay(plot,over_mslp)
 draw(plot)
 frame(wks)
end do



end
