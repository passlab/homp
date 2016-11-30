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

set object 15 rect from 0.200893, 0 to 156.248124, 10 fc rgb "#FF0000"
set object 16 rect from 0.174817, 0 to 135.613921, 10 fc rgb "#00FF00"
set object 17 rect from 0.177384, 0 to 136.156429, 10 fc rgb "#0000FF"
set object 18 rect from 0.177810, 0 to 136.764889, 10 fc rgb "#FFFF00"
set object 19 rect from 0.178621, 0 to 137.682877, 10 fc rgb "#FF00FF"
set object 20 rect from 0.179800, 0 to 138.061417, 10 fc rgb "#808080"
set object 21 rect from 0.180308, 0 to 138.525002, 10 fc rgb "#800080"
set object 22 rect from 0.181201, 0 to 138.993199, 10 fc rgb "#008080"
set object 23 rect from 0.181517, 0 to 153.435104, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.199015, 10 to 155.245001, 20 fc rgb "#FF0000"
set object 25 rect from 0.155031, 10 to 120.932916, 20 fc rgb "#00FF00"
set object 26 rect from 0.158223, 10 to 121.608013, 20 fc rgb "#0000FF"
set object 27 rect from 0.158818, 10 to 122.323714, 20 fc rgb "#FFFF00"
set object 28 rect from 0.159778, 10 to 123.307609, 20 fc rgb "#FF00FF"
set object 29 rect from 0.161073, 10 to 123.827143, 20 fc rgb "#808080"
set object 30 rect from 0.161767, 10 to 124.342064, 20 fc rgb "#800080"
set object 31 rect from 0.162705, 10 to 124.869271, 20 fc rgb "#008080"
set object 32 rect from 0.163090, 10 to 152.015943, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.197186, 20 to 185.428400, 30 fc rgb "#FF0000"
set object 34 rect from 0.092542, 20 to 72.568199, 30 fc rgb "#00FF00"
set object 35 rect from 0.095100, 20 to 73.059370, 30 fc rgb "#0000FF"
set object 36 rect from 0.095465, 20 to 73.785028, 30 fc rgb "#FFFF00"
set object 37 rect from 0.096451, 20 to 74.794225, 30 fc rgb "#FF00FF"
set object 38 rect from 0.097732, 20 to 75.136730, 30 fc rgb "#808080"
set object 39 rect from 0.098174, 20 to 107.489475, 30 fc rgb "#800080"
set object 40 rect from 0.140730, 20 to 108.151556, 30 fc rgb "#008080"
set object 41 rect from 0.141284, 20 to 150.652776, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.195381, 30 to 152.093404, 40 fc rgb "#FF0000"
set object 43 rect from 0.074500, 30 to 58.668801, 40 fc rgb "#00FF00"
set object 44 rect from 0.076978, 30 to 59.176095, 40 fc rgb "#0000FF"
set object 45 rect from 0.077343, 30 to 59.925503, 40 fc rgb "#FFFF00"
set object 46 rect from 0.078344, 30 to 60.875645, 40 fc rgb "#FF00FF"
set object 47 rect from 0.079579, 30 to 61.228928, 40 fc rgb "#808080"
set object 48 rect from 0.080052, 30 to 61.687123, 40 fc rgb "#800080"
set object 49 rect from 0.080943, 30 to 62.231184, 40 fc rgb "#008080"
set object 50 rect from 0.081348, 30 to 149.008354, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.192864, 40 to 150.252039, 50 fc rgb "#FF0000"
set object 52 rect from 0.056864, 40 to 45.581658, 50 fc rgb "#00FF00"
set object 53 rect from 0.059869, 40 to 46.076666, 50 fc rgb "#0000FF"
set object 54 rect from 0.060237, 40 to 46.813057, 50 fc rgb "#FFFF00"
set object 55 rect from 0.061237, 40 to 47.805356, 50 fc rgb "#FF00FF"
set object 56 rect from 0.062549, 40 to 48.193122, 50 fc rgb "#808080"
set object 57 rect from 0.063021, 40 to 48.665887, 50 fc rgb "#800080"
set object 58 rect from 0.063968, 40 to 49.239087, 50 fc rgb "#008080"
set object 59 rect from 0.064416, 40 to 147.276559, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.190570, 50 to 150.861138, 60 fc rgb "#FF0000"
set object 61 rect from 0.032082, 50 to 27.190381, 60 fc rgb "#00FF00"
set object 62 rect from 0.036037, 50 to 28.307596, 60 fc rgb "#0000FF"
set object 63 rect from 0.037075, 50 to 30.201806, 60 fc rgb "#FFFF00"
set object 64 rect from 0.039643, 50 to 31.761184, 60 fc rgb "#FF00FF"
set object 65 rect from 0.041577, 50 to 32.292959, 60 fc rgb "#808080"
set object 66 rect from 0.042335, 50 to 32.863875, 60 fc rgb "#800080"
set object 67 rect from 0.043513, 50 to 33.755784, 60 fc rgb "#008080"
set object 68 rect from 0.044185, 50 to 145.207648, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
