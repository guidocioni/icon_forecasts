load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

begin
; Open model level output file

fils = systemfunc ("ls /scratch/local1/m300382/icon_forecasts/*.nc") ; file paths
f_cape    = addfiles (fils, "r")  
ListSetType (f_cape, "cat")

plot_description="CAPE (colors); winds at 300 hPa (green barbs), 500 hPa (red barbs) and 850 hPa (black barbs)"

lev1=850.*100.
lev2=500.*100.
lev3=300.*100.

cape=f_cape[:]->CAPE_ML(:,0,:,:)
u=f_cape[:]->u 
v=f_cape[:]->v 
lon = f_cape[:]->lon 
lat = f_cape[:]->lat
time= f_cape[:]->time 

date_string=time/60
run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time


delete([/fils, f_cape/])

; create plot

ResC = True
ResC@gsnDraw = False
ResC@gsnFrame = False
ResC@gsnAddCyclic = False  
ResC@gsnMaximize   = True         ; Maximize plot in frame.
ResC@gsnLeftString = ""
ResV=ResC

ResC@cnFillOn = True ; do color fill
ResC@cnFillMode = "rasterfill"
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
ResC@mpNationalLineThicknessF = 4
ResC@mpDataSetName         = "Earth..4"     
; ResC@mpOutlineOn = True
ResC@tmXBLabelFontHeightF = 0.01
ResC@pmTickMarkDisplayMode = True

ResC@cnLevelSelectionMode = "ManualLevels"
ResC@cnMinLevelValF = 100
ResC@cnMaxLevelValF = 2000 
ResC@cnLevelSpacingF = 50

cmap_r              = read_colormap_file("WhBlGrYeRe")
cmap_r(0:5,3)         = 0.0    ; Fully transparent
ResC@cnFillPalette = cmap_r

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
ResV@vcWindBarbLineThicknessF = 2.0               ; change vector thickness

txres                       = True
txres@txPerimOn             = True
txres@txBackgroundFillColor = "White"
txres@txFontHeightF         = 0.009

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
   wks = gsn_open_wks("png","cape_"+date_string(time_i))

   plot = gsn_csm_contour_map(wks,cape(time_i,:,:),ResC)
   ResV@vcWindBarbColor          = "red"           ; change vector color
   over_vector=gsn_csm_vector(wks, u(time_i,{lev3},:,:), v(time_i,{lev3},:,:), ResV)
   ResV@vcWindBarbColor          = "green"           ; change vector color
   over_vector_3=gsn_csm_vector(wks, u(time_i,{lev2},:,:), v(time_i,{lev2},:,:), ResV)
   ResV@vcWindBarbColor          = "black"           ; change vector color
   over_vector_2=gsn_csm_vector(wks, u(time_i,{lev1},:,:), v(time_i,{lev1},:,:), ResV)

   text=gsn_create_text(wks, plot_description, txres)
   description=gsn_add_annotation(plot, text, amres)

   text2=gsn_create_text(wks, run_string, txres)
   run_annotation=gsn_add_annotation(plot, text2, amres2)

   overlay(plot,over_vector)
   overlay(plot,over_vector_2)
   overlay(plot,over_vector_3)

   draw(plot)
   frame(wks)
end do
end
