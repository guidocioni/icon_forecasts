load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls *.nc") ; file paths
f_rain    = addfiles (fils, "r")  
ListSetType (f_rain, "cat")

plot_description="Precipitation accumulated in the last 12 hours (mm)"

 tot_prec=f_rain[:]->tp

 lon = f_rain[:]->lon 
 lat = f_rain[:]->lat
 time= f_rain[:]->time 

 date_string=tostring(cd_calendar(time, -3))
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
 ResV=ResC

 ResC@cnFillOn = True ; do color fill
 ;ResC@cnFillMode = "rasterfill"
 ResC@cnLinesOn=False

 ResC@mpFillOn     = False
 ResC@mpLimitMode = "LatLon"                   ;
 ResC@mpMaxLonF    = 19                      ; specify the plot domain
 ResC@mpMinLonF    = 6                      ;
 ResC@mpMinLatF    = 36                     ;
 ResC@mpMaxLatF    = 48                    ;
 ResC@mpDataBaseVersion  = "MediumRes"
 ResC@mpGeophysicalLineThicknessF = 3
 ResC@mpGeophysicalLineColor ="black"
 ResC@mpOutlineBoundarySets = "AllBoundaries"     ; more outlines
 ResC@mpNationalLineThicknessF = 3
 ResC@mpDataSetName         = "Earth..4"
 ; ResC@mpOutlineOn = True
 ResC@tmXBLabelFontHeightF = 0.01
 ResC@pmTickMarkDisplayMode = True
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = 2
ResC@cnMaxLevelValF = 25 
ResC@cnLevelSpacingF = 1

txres                       = True
txres@txPerimOn             = True
txres@txBackgroundFillColor = "White"
txres@txFontHeightF         = 0.015

amres=True 
amres@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres@amOrthogonalPosF = 0.5    ; This is the bottom edge of the plot.
amres@amJust = "BottomLeft"


amres2=True 
amres2@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres2@amOrthogonalPosF = -0.5    ; This is the bottom edge of the plot.
amres2@amJust = "TopLeft"

cmap_r              = read_colormap_file("precip2_17lev")
cmap_r(0,3)         = 0.0    ; Fully transparent
ResC@cnFillPalette = cmap_r

ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = 2
ResC@cnMaxLevelValF = 80 
ResC@cnLevelSpacingF = 5

do time_i=11,dimsizes(time)-1,12
 ResC@gsnCenterString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
 wks = gsn_open_wks("png","precip_acc12h_"+date_string(time_i))
 acc_prec=tot_prec(0,:,:)
 acc_prec=0.0
 acc_prec=tot_prec(time_i,:,:)-tot_prec(time_i-11,:,:) 
 plot = gsn_csm_contour_map(wks,acc_prec,ResC)

   text=gsn_create_text(wks, plot_description, txres)
   description=gsn_add_annotation(plot, text, amres)

      text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

 draw(plot)
 frame(wks)
end do 
end
