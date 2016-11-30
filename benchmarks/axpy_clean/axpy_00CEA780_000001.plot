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

set object 15 rect from 0.132763, 0 to 162.310280, 10 fc rgb "#FF0000"
set object 16 rect from 0.110177, 0 to 131.847679, 10 fc rgb "#00FF00"
set object 17 rect from 0.112367, 0 to 132.550337, 10 fc rgb "#0000FF"
set object 18 rect from 0.112752, 0 to 133.408356, 10 fc rgb "#FFFF00"
set object 19 rect from 0.113459, 0 to 134.591220, 10 fc rgb "#FF00FF"
set object 20 rect from 0.114473, 0 to 138.129223, 10 fc rgb "#808080"
set object 21 rect from 0.117498, 0 to 138.763616, 10 fc rgb "#800080"
set object 22 rect from 0.118279, 0 to 139.407424, 10 fc rgb "#008080"
set object 23 rect from 0.118561, 0 to 155.562654, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.131152, 10 to 160.900263, 20 fc rgb "#FF0000"
set object 25 rect from 0.093637, 10 to 112.391019, 20 fc rgb "#00FF00"
set object 26 rect from 0.095855, 10 to 113.290231, 20 fc rgb "#0000FF"
set object 27 rect from 0.096379, 10 to 114.192975, 20 fc rgb "#FFFF00"
set object 28 rect from 0.097166, 10 to 115.587699, 20 fc rgb "#FF00FF"
set object 29 rect from 0.098340, 10 to 119.148063, 20 fc rgb "#808080"
set object 30 rect from 0.101354, 10 to 119.817766, 20 fc rgb "#800080"
set object 31 rect from 0.102171, 10 to 120.498059, 20 fc rgb "#008080"
set object 32 rect from 0.102502, 10 to 153.624169, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129426, 20 to 159.830385, 30 fc rgb "#FF0000"
set object 34 rect from 0.077729, 20 to 93.546389, 30 fc rgb "#00FF00"
set object 35 rect from 0.079866, 20 to 94.260817, 30 fc rgb "#0000FF"
set object 36 rect from 0.080200, 20 to 96.542980, 30 fc rgb "#FFFF00"
set object 37 rect from 0.082157, 20 to 97.973011, 30 fc rgb "#FF00FF"
set object 38 rect from 0.083356, 20 to 101.203822, 30 fc rgb "#808080"
set object 39 rect from 0.086106, 20 to 101.927666, 30 fc rgb "#800080"
set object 40 rect from 0.086965, 20 to 102.646800, 30 fc rgb "#008080"
set object 41 rect from 0.087348, 20 to 151.773956, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127913, 30 to 158.416829, 40 fc rgb "#FF0000"
set object 43 rect from 0.058995, 30 to 71.520381, 40 fc rgb "#00FF00"
set object 44 rect from 0.061197, 30 to 72.291302, 40 fc rgb "#0000FF"
set object 45 rect from 0.061534, 30 to 74.540511, 40 fc rgb "#FFFF00"
set object 46 rect from 0.063448, 30 to 76.165921, 40 fc rgb "#FF00FF"
set object 47 rect from 0.064822, 30 to 79.532085, 40 fc rgb "#808080"
set object 48 rect from 0.067708, 30 to 80.310070, 40 fc rgb "#800080"
set object 49 rect from 0.068617, 30 to 81.008018, 40 fc rgb "#008080"
set object 50 rect from 0.068950, 30 to 149.856654, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126296, 40 to 156.644303, 50 fc rgb "#FF0000"
set object 52 rect from 0.040538, 40 to 50.175845, 50 fc rgb "#00FF00"
set object 53 rect from 0.042999, 40 to 50.953827, 50 fc rgb "#0000FF"
set object 54 rect from 0.043424, 40 to 53.440788, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045539, 40 to 54.975569, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046864, 40 to 58.352326, 50 fc rgb "#808080"
set object 57 rect from 0.049693, 40 to 59.102063, 50 fc rgb "#800080"
set object 58 rect from 0.050616, 40 to 59.869454, 50 fc rgb "#008080"
set object 59 rect from 0.051014, 40 to 147.859319, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.124494, 50 to 158.330913, 60 fc rgb "#FF0000"
set object 61 rect from 0.016816, 50 to 23.091174, 60 fc rgb "#00FF00"
set object 62 rect from 0.020112, 50 to 24.685982, 60 fc rgb "#0000FF"
set object 63 rect from 0.021105, 50 to 29.100836, 60 fc rgb "#FFFF00"
set object 64 rect from 0.024882, 50 to 31.346515, 60 fc rgb "#FF00FF"
set object 65 rect from 0.026762, 50 to 35.031640, 60 fc rgb "#808080"
set object 66 rect from 0.029890, 50 to 35.956745, 60 fc rgb "#800080"
set object 67 rect from 0.031131, 50 to 37.618642, 60 fc rgb "#008080"
set object 68 rect from 0.032087, 50 to 145.311158, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
