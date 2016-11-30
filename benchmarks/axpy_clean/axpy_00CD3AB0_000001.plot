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

set object 15 rect from 0.129277, 0 to 157.870385, 10 fc rgb "#FF0000"
set object 16 rect from 0.095050, 0 to 115.820474, 10 fc rgb "#00FF00"
set object 17 rect from 0.097251, 0 to 116.713523, 10 fc rgb "#0000FF"
set object 18 rect from 0.097740, 0 to 117.568319, 10 fc rgb "#FFFF00"
set object 19 rect from 0.098455, 0 to 118.791332, 10 fc rgb "#FF00FF"
set object 20 rect from 0.099489, 0 to 119.322143, 10 fc rgb "#808080"
set object 21 rect from 0.099946, 0 to 119.997610, 10 fc rgb "#800080"
set object 22 rect from 0.100755, 0 to 120.628843, 10 fc rgb "#008080"
set object 23 rect from 0.101018, 0 to 153.825953, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.127641, 10 to 156.312629, 20 fc rgb "#FF0000"
set object 25 rect from 0.080954, 10 to 99.225505, 20 fc rgb "#00FF00"
set object 26 rect from 0.083350, 10 to 100.058780, 20 fc rgb "#0000FF"
set object 27 rect from 0.083811, 10 to 101.030735, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084658, 10 to 102.479700, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085846, 10 to 103.137235, 20 fc rgb "#808080"
set object 30 rect from 0.086422, 10 to 103.873674, 20 fc rgb "#800080"
set object 31 rect from 0.087281, 10 to 104.612503, 20 fc rgb "#008080"
set object 32 rect from 0.087627, 10 to 151.825853, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.125938, 20 to 154.042346, 30 fc rgb "#FF0000"
set object 34 rect from 0.065624, 20 to 80.550336, 30 fc rgb "#00FF00"
set object 35 rect from 0.067768, 20 to 81.397958, 30 fc rgb "#0000FF"
set object 36 rect from 0.068215, 20 to 82.434471, 30 fc rgb "#FFFF00"
set object 37 rect from 0.069071, 20 to 83.749540, 30 fc rgb "#FF00FF"
set object 38 rect from 0.070187, 20 to 84.256438, 30 fc rgb "#808080"
set object 39 rect from 0.070615, 20 to 84.982118, 30 fc rgb "#800080"
set object 40 rect from 0.071487, 20 to 85.710187, 30 fc rgb "#008080"
set object 41 rect from 0.071810, 20 to 149.899876, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.124315, 30 to 152.382962, 40 fc rgb "#FF0000"
set object 43 rect from 0.050605, 30 to 63.012105, 40 fc rgb "#00FF00"
set object 44 rect from 0.053077, 30 to 63.900373, 40 fc rgb "#0000FF"
set object 45 rect from 0.053577, 30 to 65.060025, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054539, 30 to 66.433673, 40 fc rgb "#FF00FF"
set object 47 rect from 0.055714, 30 to 67.000348, 40 fc rgb "#808080"
set object 48 rect from 0.056186, 30 to 67.708095, 40 fc rgb "#800080"
set object 49 rect from 0.057067, 30 to 68.540174, 40 fc rgb "#008080"
set object 50 rect from 0.057477, 30 to 147.892603, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.122539, 40 to 154.195369, 50 fc rgb "#FF0000"
set object 52 rect from 0.028135, 40 to 37.343164, 50 fc rgb "#00FF00"
set object 53 rect from 0.031696, 40 to 38.810062, 50 fc rgb "#0000FF"
set object 54 rect from 0.032601, 40 to 42.146750, 50 fc rgb "#FFFF00"
set object 55 rect from 0.035424, 40 to 44.451706, 50 fc rgb "#FF00FF"
set object 56 rect from 0.037311, 40 to 45.275417, 50 fc rgb "#808080"
set object 57 rect from 0.038017, 40 to 46.132601, 50 fc rgb "#800080"
set object 58 rect from 0.039297, 40 to 47.715468, 50 fc rgb "#008080"
set object 59 rect from 0.040046, 40 to 145.282791, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.130904, 50 to 160.074919, 60 fc rgb "#FF0000"
set object 61 rect from 0.109101, 50 to 132.627048, 60 fc rgb "#00FF00"
set object 62 rect from 0.111291, 50 to 133.394570, 60 fc rgb "#0000FF"
set object 63 rect from 0.111691, 50 to 134.480100, 60 fc rgb "#FFFF00"
set object 64 rect from 0.112618, 50 to 135.829838, 60 fc rgb "#FF00FF"
set object 65 rect from 0.113748, 50 to 136.384557, 60 fc rgb "#808080"
set object 66 rect from 0.114206, 50 to 137.088717, 60 fc rgb "#800080"
set object 67 rect from 0.115044, 50 to 137.855045, 60 fc rgb "#008080"
set object 68 rect from 0.115434, 50 to 155.849964, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
