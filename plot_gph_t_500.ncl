load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls /scratch/local1/m300382/icon_forecasts/*.nc") ; file paths
f_t    = addfiles (fils, "r")  
ListSetType (f_t, "cat")

plot_description="Geopotential height (dam) and temperature (deg C) at 500 hPa"

lev1=500.*100.

temp=f_t[:]->t(:,{lev1},:,:)
geopt=f_t[:]->z(:,{lev1},:,:)

lon = f_t[:]->lon 
lat = f_t[:]->lat
time= f_t[:]->time 

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time


delete([/fils, f_t/])
; conversion 

temp=temp-273.15
temp!1 = "lat"
temp!2 = "lon"
temp&lat = lat 
temp&lon = lon

geopt=geopt/100.
geopt!1 = "lat"
geopt!2 = "lon"
geopt&lat = lat 
geopt&lon = lon

; create plot

ResC = True
ResC@gsnDraw = False
ResC@gsnFrame = False
ResC@gsnAddCyclic = False  
ResC@gsnMaximize   = True         ; Maximize plot in frame.
ResC@gsnLeftString=""
ResC@gsnRightString=""

ResC2=ResC
ResC2@cnFillOn = False
ResC2@cnLinesOn= True
ResC2@cnLineLabelFontHeightF= 0.008
ResC@cnFillOn = True ; do color fill
; ResC@cnFillMode = "rasterfill"
;ResC@gsnSpreadColors=True
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
ResC@mpDataSetName         = "Earth..4"     
ResC@tmXBLabelFontHeightF = 0.01
ResC@pmTickMarkDisplayMode = True
ResC@mpNationalLineThicknessF = 3


ResC2@cnLevelSelectionMode = "ManualLevels"
ResC2@cnMinLevelValF = round(min(geopt), 0)
ResC2@cnMaxLevelValF = round(max(geopt), 0)
ResC2@cnInfoLabelOn = False  
ResC2@cnLevelSpacingF = 5
ResC2@cnLineThicknessF = 4
ResC2@cnLineColor = "white"
ResC2@cnLineLabelFontHeightF = 0.005
ResC2@cnLineLabelDensityF = 1.5
 ResC2@cnLowLabelsOn = True
 ResC2@cnLowLabelBackgroundColor = -1
ResC2@cnLowLabelString = "L"
 ResC2@cnHighLabelsOn = True
 ResC2@cnHighLabelBackgroundColor = -1
 ResC2@cnHighLabelString = "H" 

ResC@cnFillPalette = "BkBlAqGrYeOrReViWh200"
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = -50
ResC@cnMaxLevelValF = -5 
ResC@cnLevelSpacingF = 1
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

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

do time_i=0,dimsizes(time)-1
 ResC@gsnCenterString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
   wks = gsn_open_wks("png","gph_t_500_"+date_string(time_i))

   plot = gsn_csm_contour_map(wks,temp(time_i,:,:),ResC)
   over = gsn_csm_contour(wks,geopt(time_i,:,:),ResC2)

   text=gsn_create_text(wks, plot_description, txres)
   description=gsn_add_annotation(plot, text, amres)

   text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

   overlay(plot, over)
   draw(plot)
   frame(wks)
end do
end
