load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls *.nc") ; file paths
f_t    = addfiles (fils, "r")  
ListSetType (f_t, "cat")

lev1=300

plot_description="Winds (direction and intensity in km/h) and geopotential (dam) at "+lev1+" hPa"

lev1=lev1*100
geopt=f_t[:]->z(:,{lev1},:,:)
geopt=geopt/100.
u=f_t[:]->u(:,{lev1},:,:)
v=f_t[:]->v(:,{lev1},:,:)
speed=sqrt(u^2+v^2)*3.6

copy_VarCoords(u, speed)

u=mask(u, speed.le.50, False)
v=mask(v, speed.le.50, False)

lon = f_t[:]->lon 
lat = f_t[:]->lat
time= f_t[:]->time

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time

delete([/fils, f_t/])

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
;ResC@cnFillMode="RasterFill"

ResC@mpFillOn     = True
ResC@mpOutlineOn  = True 
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
ResC2@cnMinLevelValF = round(min(geopt), 0)
ResC2@cnMaxLevelValF = round(max(geopt), 0)
ResC2@cnInfoLabelOn = False  
ResC2@cnLevelSpacingF = 5
ResC2@cnLineThicknessF = 3
ResC2@cnLineColor = "grey"
ResC2@cnLineLabelFontHeightF = 0.005
ResC2@cnLineLabelDensityF = 1.2

cmap_r=read_colormap_file("wind_17lev")
cmap_r(0,:)=0.0
ResC@cnFillPalette= cmap_r
ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = 50
ResC@cnMaxLevelValF = 250 
ResC@cnLevelSpacingF = 10
ResC@lbOrientation = "Vertical"
ResC@pmLabelBarWidthF = 0.05
ResC@gsnLeftString = "ICON-EU"
ResC@gsnRightString = "Copyright DWD"

ResV@vcRefMagnitudeF           = 20             ; define vector ref mag
ResV@vcRefLengthF              = 0.02           ; define length of vec ref
ResV@vcMinDistanceF            = 0.018
ResV@vcRefAnnoOn = False 
ResV@vcGlyphStyle            = "CurlyVector"     ; turn on curly vectors
ResV@vcLineArrowColor        = "gray60"           ; change vector color
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
 wks = gsn_open_wks("png","winds_jet_"+date_string(time_i))
 
 plot = gsn_csm_contour_map(wks,speed(time_i,:,:),ResC)
 over = gsn_csm_contour(wks,geopt(time_i,:,:),ResC2)
 over_vector=gsn_csm_vector(wks, u(time_i,:,:), v(time_i,:,:), ResV)

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
