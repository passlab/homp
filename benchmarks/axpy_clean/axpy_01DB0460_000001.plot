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

set object 15 rect from 0.249484, 0 to 156.089635, 10 fc rgb "#FF0000"
set object 16 rect from 0.034991, 0 to 22.310009, 10 fc rgb "#00FF00"
set object 17 rect from 0.038730, 0 to 23.044778, 10 fc rgb "#0000FF"
set object 18 rect from 0.039659, 0 to 23.968340, 10 fc rgb "#FFFF00"
set object 19 rect from 0.041270, 0 to 24.900064, 10 fc rgb "#FF00FF"
set object 20 rect from 0.042850, 0 to 33.079272, 10 fc rgb "#808080"
set object 21 rect from 0.056913, 0 to 33.508712, 10 fc rgb "#800080"
set object 22 rect from 0.058108, 0 to 33.997008, 10 fc rgb "#008080"
set object 23 rect from 0.058466, 0 to 144.735350, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.254634, 10 to 157.366889, 20 fc rgb "#FF0000"
set object 25 rect from 0.145547, 10 to 85.753796, 20 fc rgb "#00FF00"
set object 26 rect from 0.147514, 10 to 86.130797, 20 fc rgb "#0000FF"
set object 27 rect from 0.147924, 10 to 86.511292, 20 fc rgb "#FFFF00"
set object 28 rect from 0.148619, 10 to 87.122536, 20 fc rgb "#FF00FF"
set object 29 rect from 0.149634, 10 to 94.867641, 20 fc rgb "#808080"
set object 30 rect from 0.162957, 10 to 95.189866, 20 fc rgb "#800080"
set object 31 rect from 0.163723, 10 to 95.492282, 20 fc rgb "#008080"
set object 32 rect from 0.163994, 10 to 147.996082, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.251399, 20 to 162.080265, 30 fc rgb "#FF0000"
set object 34 rect from 0.068873, 20 to 41.257898, 30 fc rgb "#00FF00"
set object 35 rect from 0.071133, 20 to 41.669860, 30 fc rgb "#0000FF"
set object 36 rect from 0.071621, 20 to 43.589819, 30 fc rgb "#FFFF00"
set object 37 rect from 0.074966, 20 to 49.057774, 30 fc rgb "#FF00FF"
set object 38 rect from 0.084329, 20 to 56.750440, 30 fc rgb "#808080"
set object 39 rect from 0.097534, 20 to 57.294670, 30 fc rgb "#800080"
set object 40 rect from 0.098697, 20 to 57.929800, 30 fc rgb "#008080"
set object 41 rect from 0.099553, 20 to 146.111080, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.257338, 30 to 167.752739, 40 fc rgb "#FF0000"
set object 43 rect from 0.209020, 30 to 122.532560, 40 fc rgb "#00FF00"
set object 44 rect from 0.210614, 30 to 122.878095, 40 fc rgb "#0000FF"
set object 45 rect from 0.210989, 30 to 124.560320, 40 fc rgb "#FFFF00"
set object 46 rect from 0.213911, 30 to 132.441189, 40 fc rgb "#FF00FF"
set object 47 rect from 0.227400, 30 to 140.255635, 40 fc rgb "#808080"
set object 48 rect from 0.240814, 30 to 140.736937, 40 fc rgb "#800080"
set object 49 rect from 0.241881, 30 to 141.064993, 40 fc rgb "#008080"
set object 50 rect from 0.242207, 30 to 149.685297, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.256023, 40 to 164.318378, 50 fc rgb "#FF0000"
set object 52 rect from 0.171256, 40 to 100.584405, 50 fc rgb "#00FF00"
set object 53 rect from 0.172958, 40 to 100.948585, 50 fc rgb "#0000FF"
set object 54 rect from 0.173354, 40 to 102.720543, 50 fc rgb "#FFFF00"
set object 55 rect from 0.176408, 40 to 107.976400, 50 fc rgb "#FF00FF"
set object 56 rect from 0.185423, 40 to 115.627111, 50 fc rgb "#808080"
set object 57 rect from 0.198568, 40 to 116.131718, 50 fc rgb "#800080"
set object 58 rect from 0.199688, 40 to 116.479002, 50 fc rgb "#008080"
set object 59 rect from 0.200013, 40 to 148.906242, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.253081, 50 to 162.991007, 60 fc rgb "#FF0000"
set object 61 rect from 0.107553, 50 to 63.532942, 60 fc rgb "#00FF00"
set object 62 rect from 0.109370, 50 to 64.048623, 60 fc rgb "#0000FF"
set object 63 rect from 0.110029, 50 to 65.849134, 60 fc rgb "#FFFF00"
set object 64 rect from 0.113147, 50 to 71.273384, 60 fc rgb "#FF00FF"
set object 65 rect from 0.122427, 50 to 78.968382, 60 fc rgb "#808080"
set object 66 rect from 0.135669, 50 to 79.482895, 60 fc rgb "#800080"
set object 67 rect from 0.136753, 50 to 79.835423, 60 fc rgb "#008080"
set object 68 rect from 0.137123, 50 to 147.129040, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
