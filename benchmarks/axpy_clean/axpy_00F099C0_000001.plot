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

set object 15 rect from 0.126943, 0 to 160.565349, 10 fc rgb "#FF0000"
set object 16 rect from 0.106183, 0 to 132.856963, 10 fc rgb "#00FF00"
set object 17 rect from 0.108096, 0 to 133.640031, 10 fc rgb "#0000FF"
set object 18 rect from 0.108518, 0 to 134.603147, 10 fc rgb "#FFFF00"
set object 19 rect from 0.109303, 0 to 135.873332, 10 fc rgb "#FF00FF"
set object 20 rect from 0.110303, 0 to 137.175574, 10 fc rgb "#808080"
set object 21 rect from 0.111367, 0 to 137.857520, 10 fc rgb "#800080"
set object 22 rect from 0.112189, 0 to 138.551806, 10 fc rgb "#008080"
set object 23 rect from 0.112475, 0 to 155.836092, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.125398, 10 to 159.010319, 20 fc rgb "#FF0000"
set object 25 rect from 0.091049, 10 to 114.210006, 20 fc rgb "#00FF00"
set object 26 rect from 0.092978, 10 to 115.032537, 20 fc rgb "#0000FF"
set object 27 rect from 0.093400, 10 to 115.982091, 20 fc rgb "#FFFF00"
set object 28 rect from 0.094209, 10 to 117.455743, 20 fc rgb "#FF00FF"
set object 29 rect from 0.095379, 10 to 118.802381, 20 fc rgb "#808080"
set object 30 rect from 0.096477, 10 to 119.560789, 20 fc rgb "#800080"
set object 31 rect from 0.097346, 10 to 120.320429, 20 fc rgb "#008080"
set object 32 rect from 0.097693, 10 to 153.824772, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.123658, 20 to 156.389784, 30 fc rgb "#FF0000"
set object 34 rect from 0.077690, 20 to 97.558337, 30 fc rgb "#00FF00"
set object 35 rect from 0.079470, 20 to 98.288382, 30 fc rgb "#0000FF"
set object 36 rect from 0.079823, 20 to 99.265059, 30 fc rgb "#FFFF00"
set object 37 rect from 0.080648, 20 to 100.659787, 30 fc rgb "#FF00FF"
set object 38 rect from 0.081746, 20 to 101.733892, 30 fc rgb "#808080"
set object 39 rect from 0.082621, 20 to 102.391178, 30 fc rgb "#800080"
set object 40 rect from 0.083407, 20 to 103.136025, 30 fc rgb "#008080"
set object 41 rect from 0.083776, 20 to 151.954036, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.122202, 30 to 154.794034, 40 fc rgb "#FF0000"
set object 43 rect from 0.063190, 30 to 79.846133, 40 fc rgb "#00FF00"
set object 44 rect from 0.065117, 30 to 80.568773, 40 fc rgb "#0000FF"
set object 45 rect from 0.065483, 30 to 81.725496, 40 fc rgb "#FFFF00"
set object 46 rect from 0.066395, 30 to 83.079530, 40 fc rgb "#FF00FF"
set object 47 rect from 0.067490, 30 to 84.169669, 40 fc rgb "#808080"
set object 48 rect from 0.068384, 30 to 84.897243, 40 fc rgb "#800080"
set object 49 rect from 0.069255, 30 to 85.663056, 40 fc rgb "#008080"
set object 50 rect from 0.069587, 30 to 150.032739, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.120632, 40 to 153.012118, 50 fc rgb "#FF0000"
set object 52 rect from 0.048212, 40 to 61.788628, 50 fc rgb "#00FF00"
set object 53 rect from 0.050484, 40 to 62.637060, 50 fc rgb "#0000FF"
set object 54 rect from 0.050912, 40 to 63.619911, 50 fc rgb "#FFFF00"
set object 55 rect from 0.051725, 40 to 65.097265, 50 fc rgb "#FF00FF"
set object 56 rect from 0.052939, 40 to 66.297145, 50 fc rgb "#808080"
set object 57 rect from 0.053879, 40 to 67.007463, 50 fc rgb "#800080"
set object 58 rect from 0.054760, 40 to 67.881787, 50 fc rgb "#008080"
set object 59 rect from 0.055181, 40 to 147.898098, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.118789, 50 to 153.384541, 60 fc rgb "#FF0000"
set object 61 rect from 0.027638, 50 to 37.303781, 60 fc rgb "#00FF00"
set object 62 rect from 0.030780, 50 to 38.681254, 60 fc rgb "#0000FF"
set object 63 rect from 0.031503, 50 to 41.011965, 60 fc rgb "#FFFF00"
set object 64 rect from 0.033432, 50 to 43.113308, 60 fc rgb "#FF00FF"
set object 65 rect from 0.035097, 50 to 44.509276, 60 fc rgb "#808080"
set object 66 rect from 0.036244, 50 to 45.415665, 60 fc rgb "#800080"
set object 67 rect from 0.037426, 50 to 46.896714, 60 fc rgb "#008080"
set object 68 rect from 0.038171, 50 to 145.335539, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
