load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls /scratch/local1/m300382/icon_forecasts/*.nc") ; file paths
f_rain    = addfiles (fils, "r")  
ListSetType (f_rain, "cat")

plot_description="Accumulated precipitation (mm) since run start"

 tot_prec=f_rain[:]->tp
 mslp=f_rain[:]->prmsl


 lon = f_rain[:]->lon 
 lat = f_rain[:]->lat
 time= f_rain[:]->time 

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time

mslp = mslp/100.

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

 ResC@cnLevelSelectionMode = "ExplicitLevels"
 ResC@cnLevels = (/5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250/)

cmap_r              = read_colormap_file("prcp_3")
cmap_r(0:1,:)=0.0
ResC@cnFillPalette = cmap_r

ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.04
ResC@pmLabelBarOrthogonalPosF = 0.
ResC@lbLabelFontHeightF = 0.005

ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResC2@cnLevelSelectionMode = "ManualLevels"
ResC2@cnMinLevelValF = round(min(mslp), 0)
ResC2@cnMaxLevelValF = round(max(mslp), 0)
ResC2@cnInfoLabelOn = False  
ResC2@cnLevelSpacingF = 4
ResC2@cnLineThicknessF = 4
ResC2@cnLineColor = "gray"
ResC2@cnLineLabelFontHeightF = 0.005

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

 do time_i=1,dimsizes(time)-1
 ResC@gsnCenterString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
 wks = gsn_open_wks("png","precip_acc_"+date_string(time_i))
 
  plot=gsn_csm_contour_map(wks, tot_prec(time_i,:,:), ResC)
  over = gsn_csm_contour(wks,mslp(time_i,:,:),ResC2)

   text=gsn_create_text(wks, plot_description, txres)
   description=gsn_add_annotation(plot, text, amres)
   text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)
   max_prec=sprintf("%3.1f", max(tot_prec(time_i,:,:)) )
   text3=gsn_create_text(wks, "maximum = "+max_prec+" mm", txres)
   extremes_annotation=gsn_add_annotation(plot, text3, amres3)

   overlay(plot, over)

 draw(plot)
 frame(wks)
end do



end
