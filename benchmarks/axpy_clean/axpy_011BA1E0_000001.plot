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

set object 15 rect from 0.212273, 0 to 156.922089, 10 fc rgb "#FF0000"
set object 16 rect from 0.133171, 0 to 95.409212, 10 fc rgb "#00FF00"
set object 17 rect from 0.135642, 0 to 95.873990, 10 fc rgb "#0000FF"
set object 18 rect from 0.136071, 0 to 96.411412, 10 fc rgb "#FFFF00"
set object 19 rect from 0.136848, 0 to 97.202027, 10 fc rgb "#FF00FF"
set object 20 rect from 0.137953, 0 to 102.780066, 10 fc rgb "#808080"
set object 21 rect from 0.145865, 0 to 103.180664, 10 fc rgb "#800080"
set object 22 rect from 0.146681, 0 to 103.572092, 10 fc rgb "#008080"
set object 23 rect from 0.146973, 0 to 149.239505, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.206298, 10 to 154.924041, 20 fc rgb "#FF0000"
set object 25 rect from 0.045050, 10 to 34.141065, 20 fc rgb "#00FF00"
set object 26 rect from 0.048917, 10 to 35.268804, 20 fc rgb "#0000FF"
set object 27 rect from 0.050140, 10 to 36.835225, 20 fc rgb "#FFFF00"
set object 28 rect from 0.052377, 10 to 37.941100, 20 fc rgb "#FF00FF"
set object 29 rect from 0.053925, 10 to 43.717322, 20 fc rgb "#808080"
set object 30 rect from 0.062205, 10 to 44.263206, 20 fc rgb "#800080"
set object 31 rect from 0.063372, 10 to 44.847881, 20 fc rgb "#008080"
set object 32 rect from 0.063795, 10 to 144.779331, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.214128, 20 to 161.464078, 30 fc rgb "#FF0000"
set object 34 rect from 0.154132, 20 to 109.937925, 30 fc rgb "#00FF00"
set object 35 rect from 0.156242, 20 to 110.355450, 30 fc rgb "#0000FF"
set object 36 rect from 0.156584, 20 to 112.582010, 30 fc rgb "#FFFF00"
set object 37 rect from 0.159742, 20 to 115.026502, 30 fc rgb "#FF00FF"
set object 38 rect from 0.163316, 20 to 120.490991, 30 fc rgb "#808080"
set object 39 rect from 0.170975, 20 to 120.948010, 30 fc rgb "#800080"
set object 40 rect from 0.171872, 20 to 121.390219, 30 fc rgb "#008080"
set object 41 rect from 0.172238, 20 to 150.547089, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.208300, 30 to 158.758633, 40 fc rgb "#FF0000"
set object 43 rect from 0.073852, 30 to 53.749891, 40 fc rgb "#00FF00"
set object 44 rect from 0.076597, 30 to 54.298597, 40 fc rgb "#0000FF"
set object 45 rect from 0.077122, 30 to 56.852407, 40 fc rgb "#FFFF00"
set object 46 rect from 0.080764, 30 to 60.145346, 40 fc rgb "#FF00FF"
set object 47 rect from 0.085436, 30 to 65.665551, 40 fc rgb "#808080"
set object 48 rect from 0.093240, 30 to 66.183226, 40 fc rgb "#800080"
set object 49 rect from 0.094274, 30 to 66.978073, 40 fc rgb "#008080"
set object 50 rect from 0.095109, 30 to 146.419806, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.215687, 40 to 162.368948, 50 fc rgb "#FF0000"
set object 52 rect from 0.180923, 40 to 128.818200, 50 fc rgb "#00FF00"
set object 53 rect from 0.183091, 40 to 129.337989, 50 fc rgb "#0000FF"
set object 54 rect from 0.183502, 40 to 131.510949, 50 fc rgb "#FFFF00"
set object 55 rect from 0.186586, 40 to 133.801689, 50 fc rgb "#FF00FF"
set object 56 rect from 0.189835, 40 to 139.167439, 50 fc rgb "#808080"
set object 57 rect from 0.197443, 40 to 139.632217, 50 fc rgb "#800080"
set object 58 rect from 0.198360, 40 to 140.051857, 50 fc rgb "#008080"
set object 59 rect from 0.198700, 40 to 151.772861, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.210426, 50 to 159.329909, 60 fc rgb "#FF0000"
set object 61 rect from 0.103657, 50 to 74.391242, 60 fc rgb "#00FF00"
set object 62 rect from 0.105932, 50 to 75.067604, 60 fc rgb "#0000FF"
set object 63 rect from 0.106554, 50 to 77.299807, 60 fc rgb "#FFFF00"
set object 64 rect from 0.109736, 50 to 79.977745, 60 fc rgb "#FF00FF"
set object 65 rect from 0.113516, 50 to 85.387927, 60 fc rgb "#808080"
set object 66 rect from 0.121198, 50 to 85.899958, 60 fc rgb "#800080"
set object 67 rect from 0.122177, 50 to 86.364736, 60 fc rgb "#008080"
set object 68 rect from 0.122577, 50 to 147.871972, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
