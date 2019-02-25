load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"

wkstype="png"
wkstype@wkWidth = 800
wkstype@wkHeight = 850

if .not.isdefined("leftlat") then 
 leftlat  =  39.54
 end if 
if .not.isdefined("rightlat") then    
 rightlat =  55.6
end if
if .not.isdefined("leftlon") then  
 leftlon  =  3.09
end if 
if .not.isdefined("rightlon") then 
 rightlon =  37.39
end if


npts     =   300                    ; number of points in resulting transect

min_pressure = 300 
max_pressure = 1000

;***********************************************
begin
;***********************************************

folder="/scratch/local1/m300382/icon_forecasts/"

fils = systemfunc ("ls "+folder+"ICON_*.nc") ; file paths
f_t    = addfiles (fils, "r")  
ListSetType (f_t, "cat")

time= f_t[:]->time 

run_time=str_sub_str(time@units,"minutes since ","")
run_string="Run: "+run_time

t=f_t[:]->t
t=t-273.15
u=f_t[:]->u
v=f_t[:]->v
r=f_t[:]->r

t&plev_3=t&plev_3/100.
u&plev_3=u&plev_3/100.
v&plev_3=v&plev_3/100.
r&plev_3=r&plev_3/100.

t2m=f_t[:]->2t(:,0,:,:)
t2m=t2m-273.15

mslp=f_t[:]->prmsl
mslp=mslp/100.
t2m=f_t[:]->2t 
t2m=t2m-273.15
gust=f_t[:]->10fg3(:,0,:,:)
gust=gust*3.6

;u=mask(u, u.lt.5 .and. u.gt.-5, False)

snow_acc=f_t[:]->SNOW_GSP+f_t[:]->SNOW_CON
rain_acc=f_t[:]->RAIN_GSP+f_t[:]->RAIN_CON

copy_VarMeta(f_t[:]->SNOW_GSP, snow_acc)
copy_VarMeta(f_t[:]->RAIN_GSP, rain_acc)

rain=rain_acc
rain=rain*0.
snow=snow_acc
snow=snow*0.

do i=1,dimsizes(time)-1
  rain(i,:,:)=rain_acc(i,:,:)-rain_acc(i-1,:,:) 
end do 

do i=1,dimsizes(time)-1
  snow(i,:,:)=snow_acc(i,:,:)-snow_acc(i-1,:,:) 
end do 

lon = f_t[:]->lon 
lat = f_t[:]->lat
time= f_t[:]->time 

;************************************
; calculate great circle along transect
;************************************

dist     = gc_latlon(leftlat,leftlon,rightlat,rightlon,npts,-4)
points   = ispan(0,npts-1,1)*1.0 

;********************************
; interpolate data to great circle
;********************************
trans_t   = linint2_points(lon,lat,t,True,dist@gclon,dist@gclat,2)
trans_u   = linint2_points(lon,lat,u,True,dist@gclon,dist@gclat,2)
trans_v   = linint2_points(lon,lat,v,True,dist@gclon,dist@gclat,2)
trans_r   = linint2_points(lon,lat,r,True,dist@gclon,dist@gclat,2)

trans_snow   = linint2_points(lon,lat,snow,True,dist@gclon,dist@gclat,2)
trans_rain   = linint2_points(lon,lat,rain,True,dist@gclon,dist@gclat,2)
trans_t2m   = linint2_points(lon,lat,t2m,True,dist@gclon,dist@gclat,2)
trans_mslp   = linint2_points(lon,lat,mslp,True,dist@gclon,dist@gclat,2)
trans_gust   = linint2_points(lon,lat,gust,True,dist@gclon,dist@gclat,2)

;oro_yaxis= dim_avg_n_Wrap(trans_oro, 1)/1000.

copy_VarAtts(t,trans_t)          ; copy attributes
copy_VarAtts(u,trans_u)          ; copy attributes
copy_VarAtts(v,trans_v)          ; copy attributes
copy_VarAtts(r,trans_r)          ; copy attributes

copy_VarCoords_2(t,trans_t)
copy_VarCoords_2(u,trans_u)          ; copy attributes
copy_VarCoords_2(v,trans_v)          ; copy attributes
copy_VarCoords_2(r,trans_r)          ; copy attributes

; compute labels for the x-axis 

nLabels      = 10
latXsecUser  = dist@gclat      ; convenience
lonXsecUser  = dist@gclon
N1           = npts-1

XBValues    = toint( fspan(0,N1,nLabels) )
XBLabels    = new(nLabels,"string")
do i=0,nLabels-1
 x = lonXsecUser(XBValues(i))
 y = latXsecUser(XBValues(i))
 XBLabels(i) = sprintf("lat %5.1f", y)+"~C~"+sprintf("lon %5.1f", x)
end do

;-----------------------------------------------------------------------
res2D = True                            ; Set basic resources 
res2D@gsnDraw              = False      ; Don't draw individual plot.
res2D@gsnFrame             = False      ; Don't advance frame.
res2D@vpWidthF             = 0.70       ; width
res2D@vpHeightF            = 0.35       ; height
res2D@tiMainFontHeightF = 0.02
res2D@gsnLeftString =  "ICON-EU"
res2D@gsnRightString = "Copyright DWD"
res2D@gsnLeftStringFontHeightF = 0.01
res2D@gsnRightStringFontHeightF = 0.01
res2D@trYReverse = True
res2D@tiYAxisString = "Pressure [pa]"
res2D@tmXBMode   = "Explicit"
res2D@tmXMajorGrid = True
res2D@tmXBValues = XBValues
res2D@tmXBMajorThicknessF = 0.4
res2D@tmXMajorGridThicknessF =0.4

tt_res = res2D
tt_res@tiMainFontHeightF =0.02
tt_res@cnFillPalette ="NCV_rainbow2" 
tt_res@cnFillOn                 = True             ; turns on color fill
tt_res@cnLevelSelectionMode     = "ManualLevels"   ; set levels manually
tt_res@cnMinLevelValF           = -40.
tt_res@cnMaxLevelValF           = 25.
tt_res@cnLevelSpacingF          = 2.
tt_res@cnLinesOn                = True
tt_res@cnLineThicknessF = 0.005
tt_res@gsnContourZeroLineThicknessF = 3
tt_res@cnLineLabelsOn           = False
tt_res@cnInfoLabelOn            = False

tt_res@lbOrientation = "Vertical"
tt_res@pmLabelBarWidthF = 0.08
tt_res@pmLabelBarHeightF = 0.35
tt_res@pmLabelBarOrthogonalPosF = -0.01
tt_res@lbLabelFontHeightF = 0.009
tt_res@lbTitleOn = True 
tt_res@lbTitleString = "[C]"
tt_res@lbTitlePosition = "Bottom"
tt_res@lbTitleFontHeightF = 0.008
tt_res@lbTitleDirection = "Across"
tt_res@lbOrientation = "Vertical"

uv_res = res2D
uv_res@vcRefAnnoOn      = False         ; turns off the ref vector
uv_res@vcRefLengthF     = 0.015         ; set length of ref vector
uv_res@vcMapDirection = False 
uv_res@vcMinDistanceF    = 0.015
uv_res@vcGlyphStyle     = "WindBarb"    ; turn on wind barbs
uv_res@vcWindBarbLineThicknessF = 4
uv_res@vcWindBarbColor = "grey60"

rh_res=res2D 
rh_res@cnFillOn=False 
rh_res@cnLinesOn = True 
rh_res@cnLevelSelectionMode = "ExplicitLevels"
rh_res@cnLevels = (/50, 70, 100/)
rh_res@cnLineThicknessF = 2
rh_res@cnLineColor = "white"
rh_res@cnInfoLabelOn = False 
rh_res@cnLineLabelFontHeightF = 0.006

mpres=True 
mpres@vpWidthF       = 0.1
mpres@vpHeightF      = 0.1
mpres@gsnDraw            = False        ; don't draw plot yet
mpres@gsnFrame           = False        ; don't advance frame yet

space                  =   2          ; arbitrary
mpres@mpMinLonF = min((/leftlon, rightlon/))-space
mpres@mpMaxLonF = max((/leftlon, rightlon/))+space
mpres@mpMinLatF = min((/leftlat, rightlat/))-space
mpres@mpMaxLatF = max((/leftlat, rightlat/))+space
mpres@mpOutlineOn = True 
mpres@mpDataBaseVersion = "MediumRes"
mpres@mpGeophysicalLineThicknessF = 1
mpres@mpGeophysicalLineColor ="black"
mpres@mpOutlineBoundarySets = "AllBoundaries"     ; more outlines
mpres@mpDataSetName         = "Earth..4"     

mpres@mpGreatCircleLinesOn = False
mpres@mpGridAndLimbOn = False 
mpres@mpOceanFillColor = "white"
mpres@mpInlandWaterFillColor = "white"
mpres@tmXBLabelsOn = False 
mpres@tmYLLabelsOn = False 
mpres@tmXBOn = False 
mpres@tmYLOn = False 

amres=True 
amres@amParallelPosF   = -0.25    ; This is the right edge of the plot.
amres@amOrthogonalPosF = -0.5    ; This is the bottom edge of the plot.
amres@amJust = "BottomLeft"

amres_plot                  = True
amres_plot@amJust           = "TopRight"  ; Corner of plot for positioning
amres_plot@amOrthogonalPosF =  -0.49   ;  0.5 is the bottom edge of the plot.
amres_plot@amParallelPosF   =  0.49    ; -0.5 is the left edge of the plot.

pres                  = True           ; polyline mods desired
pres@gsLineThicknessF = 6           ; line thickness   
pres@gsLineColor      = "blue"         ; color of line

txres                       = True
txres@txPerimOn             = True
txres@txBackgroundFillColor = "White"
txres@txFontHeightF         = 0.01


;-----------------------------------------------------------------------

res1D = True                            ; Set basic resources both will use
res1D@vpWidthF          = 0.70          ; The Width of the plot box
res1D@vpHeightF         = 0.11          ; The height of the plot box
res1D@xyLineThicknesses = (/4,4/)             ; increase line thickness
res1D@gsnDraw           = False         ; Don't draw individual plot.
res1D@gsnFrame          = False         ; Don't advance frame.
res1D@tmYLLabelFontHeightF = 0.005
res1D@trXMinF=points(0)
res1D@trXMaxF=points(npts-1)

res1D@tmXBOn = True 
res1D@tmXBMode   = "Explicit"
res1D@tmXMajorGrid = True
res1D@tmXBValues = XBValues
res1D@tmXBMajorThicknessF = 0.4
res1D@tmXMajorGridThicknessF =0.4 
res1D@tmXBLabels = XBLabels

res_xy = res1D
res_xy@trYMinF = 0.
res_xy@xyLineColors     = (/"blue","purple"/)         ; set line color
res_xy@xyDashPattern = 0
res_xy@tiYAxisString   = "Rain/snow"        ; set y-axis string
;res_xy@tmXBLabelsOn = False
res_xy@tmYLLabelsOn = False
res_xy@tmYRLabelsOn = True
res_xy@tmYLLabelDeltaF = 1.5
res_xy@tmYMajorGrid = True 
res_xy@tmYMajorGridThicknessF = 0.4

res_xy@pmLegendDisplayMode    = "Always"
res_xy@pmLegendSide           = "Bottom"               ; Change location of 
res_xy@pmLegendParallelPosF   = 0.5                 ; move units right
res_xy@pmLegendOrthogonalPosF = -1.0                ; move units down
res_xy@pmLegendWidthF         = 0.07                ; Change width and
res_xy@pmLegendHeightF        = 0.02                ; height of legend.
res_xy@lgLabelFontHeightF     = .007                 ; change font height
res_xy@lgPerimFill            = "SolidFill"      ; Fill legend box w/white
res_xy@lgPerimFillColor       = "white"          ; so it masks XY curves
res_xy@lgOrientation          = "vertical"
res_xy@lgPerimOn              = True               ; turn off box
res_xy@xyExplicitLegendLabels = (/"Rain [mm/h]","Snow [mm eq/h]"/)

t2_res = res1D
t2_res@vpHeightF         = 0.13          ; The height of the plot box
t2_res@xyLineColors      = (/"black","blue"/)        ; set line color
t2_res@tiYAxisString   = "2m Temp [C]"        ; set y-axis string
t2_res@tiYAxisSide  = "Right"
t2_res@tmXBLabelFontHeightF = 0.05
t2_res@tmXBLabelAngleF = 45
t2_res@tmYMajorGrid = True 
t2_res@tmYMajorGridThicknessF = 0.4

t2_res@tmXBLabels = XBLabels
t2_res@tiXAxisString = "Lat/Lon Extents"

slp_res = res1D
slp_res@tmYLLabelDeltaF = 2.7

wind_res= res1D
wind_res@xyLineColor = "red"
wind_res@tiYAxisString=""
wind_res@tmYRLabelFontColor = "red"
wind_res@tmYRMajorLineColor = "red"
wind_res@tmYRMinorLineColor = "red"
wind_res@tmYMajorGrid = True 
wind_res@tmYMajorGridThicknessF = 0.4
wind_res@tmYMajorGridLineColor = "red"

slp_res@xyLineColor      = "black"        ; set line color
slp_res@tiYAxisString   = "MSLP/ winds [km/h]"        ; set y-axis string
slp_res@tiYAxisSide  = "Right"
slp_res@tiYAxisOffsetXF = 0.05
;-----------------------------------------------------------------------

do time_i=50,dimsizes(time)-1
  wks = gsn_open_wks(wkstype,"/scratch/local1/m300382/icon_forecasts/cross_section_"+time_i)    ; open a workstation

  tt_res@tiMainString= "Forecast for "+cd_string(time(time_i), "%d %c. %Y")+" at "+cd_string(time(time_i), "%H:%M")+" UTC"
  ttfill    = gsn_csm_contour(wks,trans_t(time_i,{max_pressure:min_pressure},:),tt_res)
  rhcont    = gsn_csm_contour(wks, trans_r(time_i,{max_pressure:min_pressure},:), rh_res)
  windlayer = gsn_csm_vector(wks,trans_u(time_i,{max_pressure:min_pressure},:),trans_v(time_i,{max_pressure:min_pressure},:),uv_res)
  overlay(ttfill,windlayer)
  overlay(ttfill,rhcont)


  plot_description="Temperature, winds and relative humidity | "+run_string
  text=gsn_create_text(wks, plot_description, txres)
  description=gsn_add_annotation(windlayer, text, amres)

  plot_map = gsn_csm_map(wks,mpres)
  dum2 = gsn_add_polyline(wks,plot_map, dist@gclon ,dist@gclat ,pres)
  plot_overlay = gsn_add_annotation(ttfill, plot_map, amres_plot)

  rain_snow_plot= gsn_csm_xy(wks,points,(/trans_rain(time_i,:),trans_snow(time_i,:)/),res_xy)
  slp_plot  = gsn_csm_xy2(wks, points, trans_mslp(time_i,:), trans_gust(time_i,:), slp_res, wind_res)
  t2_plot  = gsn_csm_xy(wks,points,trans_t2m(time_i,:),t2_res)

  res1                     = True
  res2                     = True
  res1@gsnMaximize         = True
  res2@gsnAttachPlotsXAxis = True

  amid = gsn_attach_plots(ttfill,(/rain_snow_plot, slp_plot, t2_plot/),res1,res2)

  draw(ttfill)
  frame(wks)                            ; now frame the plot  
end do 

;-----------------------------------------------------------------------

end
