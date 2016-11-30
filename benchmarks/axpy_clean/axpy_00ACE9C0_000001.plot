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

set object 15 rect from 0.130639, 0 to 154.674982, 10 fc rgb "#FF0000"
set object 16 rect from 0.065069, 0 to 77.145201, 10 fc rgb "#00FF00"
set object 17 rect from 0.067442, 0 to 78.021446, 10 fc rgb "#0000FF"
set object 18 rect from 0.067877, 0 to 78.967927, 10 fc rgb "#FFFF00"
set object 19 rect from 0.068715, 0 to 80.213783, 10 fc rgb "#FF00FF"
set object 20 rect from 0.069792, 0 to 81.707198, 10 fc rgb "#808080"
set object 21 rect from 0.071101, 0 to 82.350853, 10 fc rgb "#800080"
set object 22 rect from 0.071897, 0 to 82.965720, 10 fc rgb "#008080"
set object 23 rect from 0.072181, 0 to 149.681195, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.128903, 10 to 153.357737, 20 fc rgb "#FF0000"
set object 25 rect from 0.048341, 10 to 58.111929, 20 fc rgb "#00FF00"
set object 26 rect from 0.050915, 10 to 59.140163, 20 fc rgb "#0000FF"
set object 27 rect from 0.051477, 10 to 60.187972, 20 fc rgb "#FFFF00"
set object 28 rect from 0.052426, 10 to 61.710173, 20 fc rgb "#FF00FF"
set object 29 rect from 0.053758, 10 to 63.316430, 20 fc rgb "#808080"
set object 30 rect from 0.055131, 10 to 64.085590, 20 fc rgb "#800080"
set object 31 rect from 0.056059, 10 to 64.835176, 20 fc rgb "#008080"
set object 32 rect from 0.056472, 10 to 147.605152, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.133923, 20 to 158.632475, 30 fc rgb "#FF0000"
set object 34 rect from 0.096931, 20 to 113.511070, 30 fc rgb "#00FF00"
set object 35 rect from 0.098949, 20 to 114.330894, 30 fc rgb "#0000FF"
set object 36 rect from 0.099406, 20 to 115.366037, 30 fc rgb "#FFFF00"
set object 37 rect from 0.100309, 20 to 116.710917, 30 fc rgb "#FF00FF"
set object 38 rect from 0.101505, 20 to 118.103005, 30 fc rgb "#808080"
set object 39 rect from 0.102696, 20 to 118.803079, 30 fc rgb "#800080"
set object 40 rect from 0.103567, 20 to 119.557271, 30 fc rgb "#008080"
set object 41 rect from 0.103966, 20 to 153.502818, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.126994, 30 to 154.238587, 40 fc rgb "#FF0000"
set object 43 rect from 0.025880, 30 to 32.962218, 40 fc rgb "#00FF00"
set object 44 rect from 0.029075, 30 to 34.279463, 40 fc rgb "#0000FF"
set object 45 rect from 0.029895, 30 to 37.167270, 40 fc rgb "#FFFF00"
set object 46 rect from 0.032452, 30 to 39.418331, 40 fc rgb "#FF00FF"
set object 47 rect from 0.034369, 30 to 41.119006, 40 fc rgb "#808080"
set object 48 rect from 0.035890, 30 to 42.041307, 40 fc rgb "#800080"
set object 49 rect from 0.037074, 30 to 43.412671, 40 fc rgb "#008080"
set object 50 rect from 0.037840, 30 to 145.195193, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.132279, 40 to 156.917982, 50 fc rgb "#FF0000"
set object 52 rect from 0.081512, 40 to 96.089812, 50 fc rgb "#00FF00"
set object 53 rect from 0.083812, 40 to 96.883152, 50 fc rgb "#0000FF"
set object 54 rect from 0.084272, 40 to 97.996594, 50 fc rgb "#FFFF00"
set object 55 rect from 0.085235, 40 to 99.389833, 50 fc rgb "#FF00FF"
set object 56 rect from 0.086460, 40 to 100.788831, 50 fc rgb "#808080"
set object 57 rect from 0.087669, 40 to 101.594838, 50 fc rgb "#800080"
set object 58 rect from 0.088605, 40 to 102.406601, 50 fc rgb "#008080"
set object 59 rect from 0.089057, 40 to 151.659366, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.135425, 50 to 160.277879, 60 fc rgb "#FF0000"
set object 61 rect from 0.113331, 50 to 132.469498, 60 fc rgb "#00FF00"
set object 62 rect from 0.115389, 50 to 133.200662, 60 fc rgb "#0000FF"
set object 63 rect from 0.115799, 50 to 134.239259, 60 fc rgb "#FFFF00"
set object 64 rect from 0.116698, 50 to 135.568018, 60 fc rgb "#FF00FF"
set object 65 rect from 0.117853, 50 to 136.926716, 60 fc rgb "#808080"
set object 66 rect from 0.119034, 50 to 137.609519, 60 fc rgb "#800080"
set object 67 rect from 0.119894, 50 to 138.329167, 60 fc rgb "#008080"
set object 68 rect from 0.120254, 50 to 155.384268, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
