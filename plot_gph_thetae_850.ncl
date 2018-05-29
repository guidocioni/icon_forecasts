load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls *.nc") ; file paths
f_t    = addfiles (fils, "r")  
ListSetType (f_t, "cat")

plot_description="~F33~q~B~~F21~e~E~ (equivalent potential temperature) and winds at 850 hPa"

lev1=850.*100.

temp_850=f_t[:]->t(:,{lev1},:,:)
geopt_850=f_t[:]->z(:,{lev1},:,:)
rh_850=f_t[:]->r(:,{lev1},:,:)
; p=f_t[:]->plev_5
u=f_t[:]->u 
v=f_t[:]->v 

lon = f_t[:]->lon 
lat = f_t[:]->lat
time= f_t[:]->time 

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time

delete([/fils, f_t/])
; conversion 

theta_e=pot_temp_equiv(lev1, temp_850, rh_850, -1, "rh")

theta_e_850=theta_e-273.15
theta_e_850!1 = "lat"
theta_e_850!2 = "lon"
theta_e_850&lat = lat 
theta_e_850&lon = lon

geopt_850=geopt_850/100.
geopt_850!1 = "lat"
geopt_850!2 = "lon"
geopt_850&lat = lat 
geopt_850&lon = lon

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
ResC@cnLinesOn=False

ResC@cnFillMode = "RasterFill"
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

ResC2@cnLevelSelectionMode = "ManualLevels"
ResC2@cnMinLevelValF = round(min(geopt_850), 0)
ResC2@cnMaxLevelValF = round(max(geopt_850), 0)
ResC2@cnInfoLabelOn = False  
ResC2@cnLevelSpacingF = 4
ResC2@cnLineThicknessF = 3
ResC2@cnLineColor = "white"
ResC2@cnLineLabelFontHeightF = 0.005
ResC2@cnLineLabelDensityF = 1.5

ResC@cnFillPalette = "BkBlAqGrYeOrReViWh200"
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = round(min(theta_e_850), 0)
ResC@cnMaxLevelValF = round(max(theta_e_850), 0) 
ResC@cnLevelSpacingF = 1
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResV@gsnLeftString = ""
ResV@gsnRightString = ""
ResV@vcRefMagnitudeF           = 5             ; define vector ref mag
ResV@vcRefLengthF              = 0.015           ; define length of vec ref
ResV@vcMinDistanceF            = 0.022
ResV@vcRefAnnoOn               = False 
ResV@vcGlyphStyle              = "WindBarb"     
ResV@vcWindBarbLineThicknessF  = 3.0               ; change vector thickness
ResV@vcWindBarbColor           = "white"

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
   wks = gsn_open_wks("png","gph_thetae_850_"+date_string(time_i))

   plot = gsn_csm_contour_map(wks,theta_e_850(time_i,:,:),ResC)
   over = gsn_csm_contour(wks,geopt_850(time_i,:,:),ResC2)
   over_vector=gsn_csm_vector(wks, u(time_i,{lev1},:,:), v(time_i,{lev1},:,:), ResV)

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
