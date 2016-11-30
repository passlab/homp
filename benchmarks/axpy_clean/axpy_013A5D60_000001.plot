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

set object 15 rect from 0.220160, 0 to 157.505568, 10 fc rgb "#FF0000"
set object 16 rect from 0.131476, 0 to 90.531288, 10 fc rgb "#00FF00"
set object 17 rect from 0.133642, 0 to 91.125662, 10 fc rgb "#0000FF"
set object 18 rect from 0.134270, 0 to 91.657541, 10 fc rgb "#FFFF00"
set object 19 rect from 0.135067, 0 to 92.392532, 10 fc rgb "#FF00FF"
set object 20 rect from 0.136144, 0 to 98.610040, 10 fc rgb "#808080"
set object 21 rect from 0.145335, 0 to 99.030525, 10 fc rgb "#800080"
set object 22 rect from 0.146188, 0 to 99.442173, 10 fc rgb "#008080"
set object 23 rect from 0.146513, 0 to 149.044380, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.214233, 10 to 155.299938, 20 fc rgb "#FF0000"
set object 25 rect from 0.036795, 10 to 27.108977, 20 fc rgb "#00FF00"
set object 26 rect from 0.040450, 10 to 28.123157, 20 fc rgb "#0000FF"
set object 27 rect from 0.041534, 10 to 29.430789, 20 fc rgb "#FFFF00"
set object 28 rect from 0.043521, 10 to 30.655540, 20 fc rgb "#FF00FF"
set object 29 rect from 0.045268, 10 to 37.125071, 20 fc rgb "#808080"
set object 30 rect from 0.054810, 10 to 37.593100, 20 fc rgb "#800080"
set object 31 rect from 0.056024, 10 to 38.198344, 20 fc rgb "#008080"
set object 32 rect from 0.056359, 10 to 144.800876, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.218321, 20 to 160.779066, 30 fc rgb "#FF0000"
set object 34 rect from 0.099520, 20 to 68.722782, 30 fc rgb "#00FF00"
set object 35 rect from 0.101532, 20 to 69.188088, 30 fc rgb "#0000FF"
set object 36 rect from 0.101974, 20 to 71.382189, 30 fc rgb "#FFFF00"
set object 37 rect from 0.105205, 20 to 75.152229, 30 fc rgb "#FF00FF"
set object 38 rect from 0.110756, 20 to 81.213502, 30 fc rgb "#808080"
set object 39 rect from 0.119731, 20 to 81.729763, 30 fc rgb "#800080"
set object 40 rect from 0.120724, 20 to 82.203906, 30 fc rgb "#008080"
set object 41 rect from 0.121139, 20 to 147.849520, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.216472, 30 to 160.701621, 40 fc rgb "#FF0000"
set object 43 rect from 0.066552, 30 to 46.510098, 40 fc rgb "#00FF00"
set object 44 rect from 0.068855, 30 to 47.152026, 40 fc rgb "#0000FF"
set object 45 rect from 0.069535, 30 to 49.649083, 40 fc rgb "#FFFF00"
set object 46 rect from 0.073254, 30 to 54.004670, 40 fc rgb "#FF00FF"
set object 47 rect from 0.079658, 30 to 60.159007, 40 fc rgb "#808080"
set object 48 rect from 0.088728, 30 to 60.759494, 40 fc rgb "#800080"
set object 49 rect from 0.089887, 30 to 61.475465, 40 fc rgb "#008080"
set object 50 rect from 0.090639, 30 to 146.504527, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.221953, 40 to 163.350845, 50 fc rgb "#FF0000"
set object 52 rect from 0.154443, 40 to 105.941585, 50 fc rgb "#00FF00"
set object 53 rect from 0.156345, 40 to 106.350520, 50 fc rgb "#0000FF"
set object 54 rect from 0.156682, 40 to 108.539863, 50 fc rgb "#FFFF00"
set object 55 rect from 0.159904, 40 to 112.454590, 50 fc rgb "#FF00FF"
set object 56 rect from 0.165681, 40 to 118.581758, 50 fc rgb "#808080"
set object 57 rect from 0.174716, 40 to 119.055223, 50 fc rgb "#800080"
set object 58 rect from 0.175653, 40 to 119.485212, 50 fc rgb "#008080"
set object 59 rect from 0.176021, 40 to 150.229058, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.223500, 50 to 164.529419, 60 fc rgb "#FF0000"
set object 61 rect from 0.185141, 50 to 126.704673, 60 fc rgb "#00FF00"
set object 62 rect from 0.186892, 50 to 127.133307, 60 fc rgb "#0000FF"
set object 63 rect from 0.187278, 50 to 129.510811, 60 fc rgb "#FFFF00"
set object 64 rect from 0.190777, 50 to 133.234653, 60 fc rgb "#FF00FF"
set object 65 rect from 0.196259, 50 to 139.456252, 60 fc rgb "#808080"
set object 66 rect from 0.205435, 50 to 139.920201, 60 fc rgb "#800080"
set object 67 rect from 0.206375, 50 to 140.335240, 60 fc rgb "#008080"
set object 68 rect from 0.206716, 50 to 151.444983, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
