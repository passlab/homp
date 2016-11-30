set title "Offloading (axpy_kernel) Profile on 6 Devices"
set yrange [0:72.000000]
set xlabel "execution time in ms"
set xrange [0:158.400000]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set border 15 lw 0.2
set xtics out nomirror
unset key
set ytics out nomirror ("dev 0(sysid:0,type:HOSTCPU)" 5,"dev 1(sysid:1,type:HOSTCPU)" 15,"dev 2(sysid:0,type:THSIM)" 25,"dev 3(sysid:1,type:THSIM)" 35,"dev 4(sysid:2,type:THSIM)" 45,"dev 5(sysid:3,type:THSIM)" 55)
set object 1 rect from 4, 65 to 17, 68 fc rgb "#FF0000"
set label "ACCU_TOTAL" at 4,63 font "Helvetica,8'"

set object 2 rect from 21, 65 to 34, 68 fc rgb "#00FF00"
set label "INIT_0" at 21,63 font "Helvetica,8'"

set object 3 rect from 38, 65 to 51, 68 fc rgb "#0000FF"
set label "INIT_0.1" at 38,63 font "Helvetica,8'"

set object 4 rect from 55, 65 to 68, 68 fc rgb "#FFFF00"
set label "INIT_1" at 55,63 font "Helvetica,8'"

set object 5 rect from 72, 65 to 85, 68 fc rgb "#00FFFF"
set label "MODELING" at 72,63 font "Helvetica,8'"

set object 6 rect from 89, 65 to 102, 68 fc rgb "#FF00FF"
set label "ACC_MAPTO" at 89,63 font "Helvetica,8'"

set object 7 rect from 106, 65 to 119, 68 fc rgb "#808080"
set label "KERN" at 106,63 font "Helvetica,8'"

set object 8 rect from 123, 65 to 136, 68 fc rgb "#800000"
set label "PRE_BAR_X" at 123,63 font "Helvetica,8'"

set object 9 rect from 140, 65 to 153, 68 fc rgb "#808000"
set label "DATA_X" at 140,63 font "Helvetica,8'"

set object 10 rect from 157, 65 to 170, 68 fc rgb "#008000"
set label "POST_BAR_X" at 157,63 font "Helvetica,8'"

set object 11 rect from 174, 65 to 187, 68 fc rgb "#800080"
set label "ACC_MAPFROM" at 174,63 font "Helvetica,8'"

set object 12 rect from 191, 65 to 204, 68 fc rgb "#008080"
set label "FINI_1" at 191,63 font "Helvetica,8'"

set object 13 rect from 208, 65 to 221, 68 fc rgb "#000080"
set label "BAR_FINI_2" at 208,63 font "Helvetica,8'"

set object 14 rect from 225, 65 to 238, 68 fc rgb "(null)"
set label "PROF_BAR" at 225,63 font "Helvetica,8'"

set object 15 rect from 1.017592, 0 to 170.641245, 10 fc rgb "#FF0000"
set object 16 rect from 0.411265, 0 to 62.922903, 10 fc rgb "#00FF00"
set object 17 rect from 0.426477, 0 to 63.664529, 10 fc rgb "#0000FF"
set object 18 rect from 0.429381, 0 to 64.505674, 10 fc rgb "#FFFF00"
set object 19 rect from 0.435348, 0 to 65.881835, 10 fc rgb "#FF00FF"
set object 20 rect from 0.444587, 0 to 82.588129, 10 fc rgb "#808080"
set object 21 rect from 0.558605, 0 to 83.537107, 10 fc rgb "#800080"
set object 22 rect from 0.565201, 0 to 84.158424, 10 fc rgb "#008080"
set object 23 rect from 0.567363, 0 to 150.477243, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.027412, 10 to 158.073685, 20 fc rgb "#FF0000"
set object 25 rect from 0.766926, 10 to 115.380142, 20 fc rgb "#00FF00"
set object 26 rect from 0.779066, 10 to 115.989424, 20 fc rgb "#0000FF"
set object 27 rect from 0.781830, 10 to 116.715159, 20 fc rgb "#FFFF00"
set object 28 rect from 0.786617, 10 to 117.772716, 20 fc rgb "#FF00FF"
set object 29 rect from 0.793759, 10 to 120.971977, 20 fc rgb "#808080"
set object 30 rect from 0.815348, 10 to 121.597897, 20 fc rgb "#800080"
set object 31 rect from 0.821092, 10 to 122.126082, 20 fc rgb "#008080"
set object 32 rect from 0.823032, 10 to 152.088534, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.996095, 20 to 154.698117, 30 fc rgb "#FF0000"
set object 34 rect from 0.292081, 20 to 45.384079, 30 fc rgb "#00FF00"
set object 35 rect from 0.307854, 20 to 46.050251, 30 fc rgb "#0000FF"
set object 36 rect from 0.310776, 20 to 47.868002, 30 fc rgb "#FFFF00"
set object 37 rect from 0.323029, 20 to 49.261840, 30 fc rgb "#FF00FF"
set object 38 rect from 0.332473, 20 to 52.258799, 30 fc rgb "#808080"
set object 39 rect from 0.352692, 20 to 52.830504, 30 fc rgb "#800080"
set object 40 rect from 0.358262, 20 to 53.486575, 30 fc rgb "#008080"
set object 41 rect from 0.360939, 20 to 147.312444, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.983445, 30 to 154.958495, 40 fc rgb "#FF0000"
set object 43 rect from 0.138670, 30 to 23.412574, 40 fc rgb "#00FF00"
set object 44 rect from 0.160962, 30 to 24.479638, 40 fc rgb "#0000FF"
set object 45 rect from 0.165655, 30 to 27.249638, 40 fc rgb "#FFFF00"
set object 46 rect from 0.184510, 30 to 29.319303, 40 fc rgb "#FF00FF"
set object 47 rect from 0.198228, 30 to 32.480240, 40 fc rgb "#808080"
set object 48 rect from 0.219555, 30 to 33.205233, 40 fc rgb "#800080"
set object 49 rect from 0.227377, 30 to 34.348495, 40 fc rgb "#008080"
set object 50 rect from 0.232137, 30 to 145.175791, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.037009, 40 to 160.574542, 50 fc rgb "#FF0000"
set object 52 rect from 0.873218, 40 to 131.223008, 50 fc rgb "#00FF00"
set object 53 rect from 0.885894, 40 to 131.837789, 50 fc rgb "#0000FF"
set object 54 rect from 0.888358, 40 to 133.643064, 50 fc rgb "#FFFF00"
set object 55 rect from 0.900664, 40 to 134.933374, 50 fc rgb "#FF00FF"
set object 56 rect from 0.909209, 40 to 137.912806, 50 fc rgb "#808080"
set object 57 rect from 0.929496, 40 to 138.526247, 50 fc rgb "#800080"
set object 58 rect from 0.935182, 40 to 139.149642, 50 fc rgb "#008080"
set object 59 rect from 0.937635, 40 to 153.552479, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.006413, 50 to 156.419171, 60 fc rgb "#FF0000"
set object 61 rect from 0.653558, 50 to 98.948636, 60 fc rgb "#00FF00"
set object 62 rect from 0.668678, 50 to 99.611837, 60 fc rgb "#0000FF"
set object 63 rect from 0.671409, 50 to 101.499547, 60 fc rgb "#FFFF00"
set object 64 rect from 0.684115, 50 to 102.959333, 60 fc rgb "#FF00FF"
set object 65 rect from 0.694116, 50 to 106.032191, 60 fc rgb "#808080"
set object 66 rect from 0.714711, 50 to 106.630337, 60 fc rgb "#800080"
set object 67 rect from 0.720275, 50 to 107.317898, 60 fc rgb "#008080"
set object 68 rect from 0.723306, 50 to 148.961610, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
