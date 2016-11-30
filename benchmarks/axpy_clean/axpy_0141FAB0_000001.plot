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

set object 15 rect from 0.126745, 0 to 159.940376, 10 fc rgb "#FF0000"
set object 16 rect from 0.106120, 0 to 133.589602, 10 fc rgb "#00FF00"
set object 17 rect from 0.108366, 0 to 134.304477, 10 fc rgb "#0000FF"
set object 18 rect from 0.108698, 0 to 135.182644, 10 fc rgb "#FFFF00"
set object 19 rect from 0.109422, 0 to 136.466426, 10 fc rgb "#FF00FF"
set object 20 rect from 0.110454, 0 to 136.988373, 10 fc rgb "#808080"
set object 21 rect from 0.110878, 0 to 137.655034, 10 fc rgb "#800080"
set object 22 rect from 0.111700, 0 to 138.334080, 10 fc rgb "#008080"
set object 23 rect from 0.111960, 0 to 156.103701, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.125184, 10 to 158.584790, 20 fc rgb "#FF0000"
set object 25 rect from 0.090873, 10 to 114.716669, 20 fc rgb "#00FF00"
set object 26 rect from 0.093104, 10 to 115.678952, 20 fc rgb "#0000FF"
set object 27 rect from 0.093642, 10 to 116.654800, 20 fc rgb "#FFFF00"
set object 28 rect from 0.094473, 10 to 118.151342, 20 fc rgb "#FF00FF"
set object 29 rect from 0.095668, 10 to 118.733887, 20 fc rgb "#808080"
set object 30 rect from 0.096151, 10 to 119.466086, 20 fc rgb "#800080"
set object 31 rect from 0.096978, 10 to 120.208238, 20 fc rgb "#008080"
set object 32 rect from 0.097322, 10 to 154.113670, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.123552, 20 to 156.164374, 30 fc rgb "#FF0000"
set object 34 rect from 0.077036, 20 to 97.514701, 30 fc rgb "#00FF00"
set object 35 rect from 0.079185, 20 to 98.212325, 30 fc rgb "#0000FF"
set object 36 rect from 0.079518, 20 to 99.246340, 30 fc rgb "#FFFF00"
set object 37 rect from 0.080358, 20 to 100.542506, 30 fc rgb "#FF00FF"
set object 38 rect from 0.081401, 20 to 101.053321, 30 fc rgb "#808080"
set object 39 rect from 0.081826, 20 to 101.722489, 30 fc rgb "#800080"
set object 40 rect from 0.082630, 20 to 102.459627, 30 fc rgb "#008080"
set object 41 rect from 0.082956, 20 to 152.200235, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.121924, 30 to 154.321343, 40 fc rgb "#FF0000"
set object 43 rect from 0.061430, 30 to 78.175555, 40 fc rgb "#00FF00"
set object 44 rect from 0.063646, 30 to 78.985678, 40 fc rgb "#0000FF"
set object 45 rect from 0.063974, 30 to 80.118626, 40 fc rgb "#FFFF00"
set object 46 rect from 0.064900, 30 to 81.468019, 40 fc rgb "#FF00FF"
set object 47 rect from 0.065983, 30 to 82.003530, 40 fc rgb "#808080"
set object 48 rect from 0.066417, 30 to 82.682577, 40 fc rgb "#800080"
set object 49 rect from 0.067234, 30 to 83.414776, 40 fc rgb "#008080"
set object 50 rect from 0.067558, 30 to 150.117389, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.120332, 40 to 152.467401, 50 fc rgb "#FF0000"
set object 52 rect from 0.046575, 40 to 60.091734, 50 fc rgb "#00FF00"
set object 53 rect from 0.048983, 40 to 60.916748, 50 fc rgb "#0000FF"
set object 54 rect from 0.049381, 40 to 62.044683, 50 fc rgb "#FFFF00"
set object 55 rect from 0.050281, 40 to 63.447304, 50 fc rgb "#FF00FF"
set object 56 rect from 0.051429, 40 to 64.061991, 50 fc rgb "#808080"
set object 57 rect from 0.051930, 40 to 64.712581, 50 fc rgb "#800080"
set object 58 rect from 0.052721, 40 to 65.527643, 50 fc rgb "#008080"
set object 59 rect from 0.053124, 40 to 148.051867, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.118442, 50 to 153.016477, 60 fc rgb "#FF0000"
set object 61 rect from 0.025619, 50 to 35.398287, 60 fc rgb "#00FF00"
set object 62 rect from 0.029485, 50 to 37.445231, 60 fc rgb "#0000FF"
set object 63 rect from 0.030437, 50 to 39.880613, 60 fc rgb "#FFFF00"
set object 64 rect from 0.032451, 50 to 41.901607, 60 fc rgb "#FF00FF"
set object 65 rect from 0.033999, 50 to 42.698091, 60 fc rgb "#808080"
set object 66 rect from 0.034664, 50 to 43.591076, 60 fc rgb "#800080"
set object 67 rect from 0.035928, 50 to 45.095064, 60 fc rgb "#008080"
set object 68 rect from 0.036597, 50 to 145.256652, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
