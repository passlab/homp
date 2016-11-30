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

set object 15 rect from 0.111631, 0 to 162.612419, 10 fc rgb "#FF0000"
set object 16 rect from 0.092204, 0 to 132.674511, 10 fc rgb "#00FF00"
set object 17 rect from 0.094071, 0 to 133.542539, 10 fc rgb "#0000FF"
set object 18 rect from 0.094427, 0 to 134.530936, 10 fc rgb "#FFFF00"
set object 19 rect from 0.095125, 0 to 135.973883, 10 fc rgb "#FF00FF"
set object 20 rect from 0.096145, 0 to 137.480553, 10 fc rgb "#808080"
set object 21 rect from 0.097208, 0 to 138.257954, 10 fc rgb "#800080"
set object 22 rect from 0.098037, 0 to 139.029700, 10 fc rgb "#008080"
set object 23 rect from 0.098308, 0 to 157.299454, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.110116, 10 to 160.937259, 20 fc rgb "#FF0000"
set object 25 rect from 0.078340, 10 to 113.279005, 20 fc rgb "#00FF00"
set object 26 rect from 0.080371, 10 to 114.220675, 20 fc rgb "#0000FF"
set object 27 rect from 0.080792, 10 to 115.177916, 20 fc rgb "#FFFF00"
set object 28 rect from 0.081512, 10 to 116.930969, 20 fc rgb "#FF00FF"
set object 29 rect from 0.082736, 10 to 118.603311, 20 fc rgb "#808080"
set object 30 rect from 0.083937, 10 to 119.560563, 20 fc rgb "#800080"
set object 31 rect from 0.084838, 10 to 120.451253, 20 fc rgb "#008080"
set object 32 rect from 0.085196, 10 to 154.940339, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.108351, 20 to 158.191569, 30 fc rgb "#FF0000"
set object 34 rect from 0.065613, 20 to 94.829410, 30 fc rgb "#00FF00"
set object 35 rect from 0.067359, 20 to 95.707355, 30 fc rgb "#0000FF"
set object 36 rect from 0.067708, 20 to 96.840186, 30 fc rgb "#FFFF00"
set object 37 rect from 0.068524, 20 to 98.529525, 30 fc rgb "#FF00FF"
set object 38 rect from 0.069714, 20 to 99.886096, 30 fc rgb "#808080"
set object 39 rect from 0.070659, 20 to 100.673414, 30 fc rgb "#800080"
set object 40 rect from 0.071498, 20 to 101.589583, 30 fc rgb "#008080"
set object 41 rect from 0.071883, 20 to 152.800701, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.106919, 30 to 156.107141, 40 fc rgb "#FF0000"
set object 43 rect from 0.051443, 30 to 74.898633, 40 fc rgb "#00FF00"
set object 44 rect from 0.053271, 30 to 75.765258, 40 fc rgb "#0000FF"
set object 45 rect from 0.053624, 30 to 77.011367, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054536, 30 to 78.638396, 40 fc rgb "#FF00FF"
set object 47 rect from 0.055660, 30 to 79.881689, 40 fc rgb "#808080"
set object 48 rect from 0.056546, 30 to 80.721400, 40 fc rgb "#800080"
set object 49 rect from 0.057400, 30 to 81.621996, 40 fc rgb "#008080"
set object 50 rect from 0.057770, 30 to 150.587422, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.105408, 40 to 154.279055, 50 fc rgb "#FF0000"
set object 52 rect from 0.037365, 40 to 55.477638, 50 fc rgb "#00FF00"
set object 53 rect from 0.039545, 40 to 56.392404, 50 fc rgb "#0000FF"
set object 54 rect from 0.039944, 40 to 57.535151, 50 fc rgb "#FFFF00"
set object 55 rect from 0.040765, 40 to 59.296707, 50 fc rgb "#FF00FF"
set object 56 rect from 0.042007, 40 to 60.670265, 50 fc rgb "#808080"
set object 57 rect from 0.042965, 40 to 61.562369, 50 fc rgb "#800080"
set object 58 rect from 0.043887, 40 to 62.484214, 50 fc rgb "#008080"
set object 59 rect from 0.044269, 40 to 148.385485, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.103647, 50 to 154.984228, 60 fc rgb "#FF0000"
set object 61 rect from 0.018586, 50 to 29.862884, 60 fc rgb "#00FF00"
set object 62 rect from 0.021587, 50 to 31.433268, 60 fc rgb "#0000FF"
set object 63 rect from 0.022333, 50 to 34.389965, 60 fc rgb "#FFFF00"
set object 64 rect from 0.024450, 50 to 36.736347, 60 fc rgb "#FF00FF"
set object 65 rect from 0.026073, 50 to 38.378948, 60 fc rgb "#808080"
set object 66 rect from 0.027254, 50 to 39.404165, 60 fc rgb "#800080"
set object 67 rect from 0.028399, 50 to 41.127487, 60 fc rgb "#008080"
set object 68 rect from 0.029191, 50 to 145.353744, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
