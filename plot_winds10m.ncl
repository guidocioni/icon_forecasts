load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls *.nc") ; file paths
f_pres    = addfiles (fils, "r")  
ListSetType (f_pres, "cat")

plot_description="MSLP (red contours, hPa), winds (direction and intensity in km/h) at 10m"

vmax_10m=f_pres[:]->10fg3
u_10m=f_pres[:]->10u
v_10m=f_pres[:]->10v
mslp=f_pres[:]->prmsl

lon = f_pres[:]->lon 
lat = f_pres[:]->lat
time= f_pres[:]->time

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
 run_string="Run: "+run_time


delete([/fils, f_pres/])

; conversion 

mslp = mslp/100.

; vmax_10m=wind_speed(u_10m, v_10m)
vmax_10m=vmax_10m*3.6

; create plot

ResC = True
ResC@gsnDraw = False
ResC@gsnFrame = False
ResC@gsnAddCyclic = False  
ResC@gsnMaximize   = True         ; Maximize plot in frame.
ResC@gsnLeftString=""
ResC@gsnRightString=""

ResV=ResC
ResC2=ResC

ResC2@cnFillOn = False
ResC2@cnLinesOn= True
ResC2@cnLineLabelFontHeightF= 0.008
ResC@cnFillOn = True ; do color fill
; ResC@cnFillMode = "rasterfill"
;ResC@gsnSpreadColors=True
ResC@cnLinesOn=False
ResC@cnFillMode="RasterFill"

; ResC@mpFillOn = True
ResC@mpOutlineOn = True 
ResC@mpLimitMode = "LatLon"                   ;
ResC@mpMaxLonF    = max(lon)                      ; specify the plot domain
ResC@mpMinLonF    = min(lon)                      ;
ResC@mpMinLatF    = min(lat)                     ;
ResC@mpMaxLatF    = max(lat)                    ;
ResC@mpDataBaseVersion  = "MediumRes"
ResC@mpGeophysicalLineThicknessF = 2
ResC@mpGeophysicalLineColor ="black"
ResC@mpOutlineBoundarySets = "AllBoundaries"     ; more outlines
ResC@mpDataSetName         = "Earth..4"
ResC@mpNationalLineThicknessF = 3

; ResC@mpOutlineOn = True
ResC@tmXBLabelFontHeightF = 0.01
ResC@pmTickMarkDisplayMode = True
ResC2@cnLevelSelectionMode = "ManualLevels"
ResC2@cnMinLevelValF = round(min(mslp), 0)
ResC2@cnMaxLevelValF = round(max(mslp), 0)
ResC2@cnInfoLabelOn = False  
ResC2@cnLevelSpacingF = 4
ResC2@cnLineThicknessF = 3
ResC2@cnLineColor = "red"
ResC2@cnLineLabelFontHeightF = 0.005
ResC2@cnLineLabelDensityF = 2


cmap_r=read_colormap_file("wind_17lev")
cmap_r(0,:)=0.0
ResC@cnFillPalette= cmap_r
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = 15
ResC@cnMaxLevelValF = 115 
ResC@cnLevelSpacingF = 5
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResV@vcRefMagnitudeF           = 10             ; define vector ref mag
ResV@vcRefLengthF              = 0.03           ; define length of vec ref
ResV@vcMinDistanceF            = 0.015
ResV@vcRefAnnoOn = False 
ResV@vcGlyphStyle            = "CurlyVector"     ; turn on curly vectors
ResV@vcLineArrowColor        = "gray40"           ; change vector color
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
 wks = gsn_open_wks("png","winds10m_"+date_string(time_i))
 
 plot = gsn_csm_contour_map(wks,vmax_10m(time_i,0,:,:),ResC)
 over = gsn_csm_contour(wks,mslp(time_i,:,:),ResC2)
 over_vector=gsn_csm_vector(wks, u_10m(time_i,0,:,:), v_10m(time_i,0,:,:), ResV)

 text=gsn_create_text(wks, plot_description, txres)
 description=gsn_add_annotation(plot, text, amres)

    text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

 overlay(plot,over_vector)
 overlay(plot, over)
 draw(plot)
 frame(wks)
end do
end
