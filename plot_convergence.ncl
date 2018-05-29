load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls /scratch/local1/m300382/icon_forecasts/*.nc") ; file paths
f_pres    = addfiles (fils, "r")  
ListSetType (f_pres, "cat")


u_10m=f_pres[:]->10u
v_10m=f_pres[:]->10v

lon = f_pres[:]->lon 
lat = f_pres[:]->lat
time= f_pres[:]->time

plot_description="Winds (vectors) and convergence (colors, s~S~-1~N~ ) at 10 m"

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
 run_string="Run: "+run_time


delete([/fils, f_pres/])

; conversion 

conv=uv2dv_cfd(u_10m, v_10m, lat, lon, 0)
conv=-conv

copy_VarCoords(u_10m, conv)

; create plot

ResC = True
ResC@gsnDraw = False
ResC@gsnFrame = False
ResC@gsnAddCyclic = False  
ResC@gsnMaximize   = True         ; Maximize plot in frame.
ResC@gsnLeftString=""
ResC@gsnRightString=""

ResV=ResC
ResC@cnFillOn = True ; do color fill
ResC@cnLinesOn=False
ResC@cnFillMode = "RasterFill"

ResC@mpFillOn     = False
ResC@mpLimitMode = "LatLon"                   ;
ResC@mpMaxLonF    = max(lon)                      ; specify the plot domain
ResC@mpMinLonF    = min(lon)                      ;
ResC@mpMinLatF    = min(lat)                     ;
ResC@mpMaxLatF    = max(lat)                     ;
ResC@mpDataBaseVersion  = "MediumRes"
ResC@mpGeophysicalLineThicknessF = 2
ResC@mpGeophysicalLineColor ="black"
ResC@mpOutlineBoundarySets = "AllBoundaries"     ; more outlines
ResC@mpDataSetName         = "Earth..4"
ResC@mpNationalLineThicknessF = 2     
ResC@tmXBLabelFontHeightF = 0.01
ResC@pmTickMarkDisplayMode = True

ResC@cnFillPalette = "MPL_PuOr"
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = -0.0005
ResC@cnMaxLevelValF = 0.0005
ResC@cnLevelSpacingF = 0.00005
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResV@vcRefMagnitudeF           = 10             ; define vector ref mag
ResV@vcRefLengthF              = 0.03           ; define length of vec ref
ResV@vcMinDistanceF            = 0.015
ResV@vcRefAnnoOn = False 
ResV@vcGlyphStyle            = "CurlyVector"     ; turn on curly vectors
ResV@vcLineArrowColor        = "grey60"           ; change vector color
ResV@vcLineArrowThicknessF   = 2.0               ; change vector thickness

txres                       = True
txres@txPerimOn             = True
txres@txBackgroundFillColor = "White"
txres@txFontHeightF         = 0.012

amres=True 
amres@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres@amOrthogonalPosF = 0.5    ; This is the bottom edge of the plot.
amres@amJust = "BottomLeft"

amres2=True 
amres2@amParallelPosF   = -0.5    ; This is the right edge of the plot.
amres2@amOrthogonalPosF = -0.5    ; This is the bottom edge of the plot.
amres2@amJust = "TopLeft"

do time_i=0,dimsizes(time)-1
 ResC@gsnCenterString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
   wks = gsn_open_wks("png","conv_"+date_string(time_i))
   
   plot = gsn_csm_contour_map(wks,conv(time_i,0,:,:),ResC)
   over_vector=gsn_csm_vector(wks, u_10m(time_i,0,:,:), v_10m(time_i,0,:,:), ResV)

   text=gsn_create_text(wks, plot_description, txres)
   description=gsn_add_annotation(plot, text, amres)

   text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

   overlay(plot,over_vector)
   draw(plot)
   frame(wks)
end do
end
