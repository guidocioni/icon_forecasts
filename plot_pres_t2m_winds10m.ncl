load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls ICON_2017092812_eur.nc") ; file paths
f_pres    = addfiles (fils, "r")  
ListSetType (f_pres, "cat")

plot_description="MSLP (white contours, hPa), temperature at 2m (deg C, colors) and 10m winds (vectors)"


mslp=f_pres[:]->prmsl
t_2m=f_pres[:]->2t
u_10m=f_pres[:]->10u
v_10m=f_pres[:]->10v

lon = f_pres[:]->lon 
lat = f_pres[:]->lat
time= f_pres[:]->time

date_string=tostring(cd_calendar(time, -3))
run_time=str_sub_str(time@units,"minutes since ","")
 run_string="Run: "+run_time


delete([/fils, f_pres/])

; conversion 

mslp = mslp/100.
; mslp = smth9_Wrap(mslp, 0.5, -0.25, False)
t_2m=t_2m-273.15

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
; ResC2@cnFillDrawOrder = "PreDraw"

ResC2@cnFillOn = False
ResC2@cnLinesOn= True
ResC2@cnLineLabelFontHeightF= 0.008
ResC@cnFillOn = True ; do color fill
; ResC@cnFillMode = "rasterfill"
;ResC@gsnSpreadColors=True
ResC@cnLinesOn=False

ResC@cnFillMode ="RasterFill"
ResC@mpFillOn     = False
ResC@mpLimitMode = "LatLon"                   ;
ResC@mpMaxLonF    = max(lon)                      ; specify the plot domain
ResC@mpMinLonF    = min(lon)                      ;
ResC@mpMinLatF    = min(lat)                     ;
ResC@mpMaxLatF    = max(lat)                    ;
ResC@mpDataBaseVersion  = "MediumRes"
ResC@mpGeophysicalLineThicknessF = 4
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
ResC2@cnLineThicknessF = 4
ResC2@cnLineColor = "white"
ResC2@cnLineLabelFontHeightF = 0.005
ResC2@cnLowLabelsOn = True
ResC2@cnLowLabelBackgroundColor = -1 
ResC2@cnHighLabelsOn = True
ResC2@cnHighLabelBackgroundColor = -1
ResC2@cnLowLabelFontColor = "white" 
ResC2@cnHighLabelFontColor = "white" 

ResC@cnFillPalette = "BkBlAqGrYeOrReViWh200"
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = round(min(t_2m), 0)
ResC@cnMaxLevelValF = round(max(t_2m), 0) 
ResC@cnLevelSpacingF = 1
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResV@vcRefMagnitudeF           = 10             ; define vector ref mag
ResV@vcRefLengthF              = 0.04           ; define length of vec ref
ResV@vcMinDistanceF            = 0.012
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
 wks = gsn_open_wks("x11","t_v_pres_"+date_string(time_i))
 
 plot = gsn_csm_contour_map(wks,t_2m(time_i,0,:,:),ResC)
 over = gsn_csm_contour(wks,mslp(time_i,:,:),ResC2)
 over_vector=gsn_csm_vector(wks, u_10m(time_i,0,:,:), v_10m(time_i,0,:,:), ResV)

 text=gsn_create_text(wks, plot_description, txres)
 description=gsn_add_annotation(plot, text, amres)

    text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

 overlay(plot, over)
 overlay(plot,over_vector)
 draw(plot)
 frame(wks)
end do
end
