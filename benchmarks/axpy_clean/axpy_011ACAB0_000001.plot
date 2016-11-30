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

set object 15 rect from 1.559137, 0 to 144.107490, 10 fc rgb "#FF0000"
set object 16 rect from 1.547115, 0 to 142.914697, 10 fc rgb "#00FF00"
set object 17 rect from 1.549326, 0 to 142.980574, 10 fc rgb "#0000FF"
set object 18 rect from 1.549810, 0 to 143.049497, 10 fc rgb "#FFFF00"
set object 19 rect from 1.550552, 0 to 143.144344, 10 fc rgb "#FF00FF"
set object 20 rect from 1.551579, 0 to 143.181525, 10 fc rgb "#808080"
set object 21 rect from 1.551989, 0 to 143.233010, 10 fc rgb "#800080"
set object 22 rect from 1.552810, 0 to 143.285139, 10 fc rgb "#008080"
set object 23 rect from 1.553132, 0 to 143.740183, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.572507, 10 to 145.402962, 20 fc rgb "#FF0000"
set object 25 rect from 1.529085, 10 to 141.268525, 20 fc rgb "#00FF00"
set object 26 rect from 1.531495, 10 to 141.351747, 20 fc rgb "#0000FF"
set object 27 rect from 1.532154, 10 to 141.424636, 20 fc rgb "#FFFF00"
set object 28 rect from 1.532993, 10 to 141.539875, 20 fc rgb "#FF00FF"
set object 29 rect from 1.534224, 10 to 141.605933, 20 fc rgb "#808080"
set object 30 rect from 1.534946, 10 to 141.662307, 20 fc rgb "#800080"
set object 31 rect from 1.535784, 10 to 141.717388, 20 fc rgb "#008080"
set object 32 rect from 1.536122, 10 to 145.029567, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.570825, 20 to 145.368267, 30 fc rgb "#FF0000"
set object 34 rect from 1.509177, 20 to 139.485711, 30 fc rgb "#00FF00"
set object 35 rect from 1.512183, 20 to 139.580652, 30 fc rgb "#0000FF"
set object 36 rect from 1.512966, 20 to 139.752078, 30 fc rgb "#FFFF00"
set object 37 rect from 1.514876, 20 to 139.889271, 30 fc rgb "#FF00FF"
set object 38 rect from 1.516314, 20 to 139.945276, 30 fc rgb "#808080"
set object 39 rect from 1.516916, 20 to 139.997773, 30 fc rgb "#800080"
set object 40 rect from 1.517791, 20 to 140.065591, 30 fc rgb "#008080"
set object 41 rect from 1.518225, 20 to 144.884621, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.569232, 30 to 145.061029, 40 fc rgb "#FF0000"
set object 43 rect from 0.058099, 30 to 5.534171, 40 fc rgb "#00FF00"
set object 44 rect from 0.060452, 30 to 5.601797, 40 fc rgb "#0000FF"
set object 45 rect from 0.060829, 30 to 5.683545, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061718, 30 to 5.789187, 40 fc rgb "#FF00FF"
set object 47 rect from 0.062861, 30 to 5.835228, 40 fc rgb "#808080"
set object 48 rect from 0.063375, 30 to 5.888368, 40 fc rgb "#800080"
set object 49 rect from 0.064227, 30 to 5.945204, 40 fc rgb "#008080"
set object 50 rect from 0.064560, 30 to 144.702587, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.566305, 40 to 144.794118, 50 fc rgb "#FF0000"
set object 52 rect from 0.042620, 40 to 4.129911, 50 fc rgb "#00FF00"
set object 53 rect from 0.045226, 40 to 4.194869, 50 fc rgb "#0000FF"
set object 54 rect from 0.045583, 40 to 4.276150, 50 fc rgb "#FFFF00"
set object 55 rect from 0.046470, 40 to 4.387424, 50 fc rgb "#FF00FF"
set object 56 rect from 0.047666, 40 to 4.430324, 50 fc rgb "#808080"
set object 57 rect from 0.048168, 40 to 4.489096, 50 fc rgb "#800080"
set object 58 rect from 0.049062, 40 to 4.546300, 50 fc rgb "#008080"
set object 59 rect from 0.049425, 40 to 144.426348, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.563515, 50 to 144.866451, 60 fc rgb "#FF0000"
set object 61 rect from 0.020714, 50 to 2.176595, 60 fc rgb "#00FF00"
set object 62 rect from 0.024217, 50 to 2.303828, 60 fc rgb "#0000FF"
set object 63 rect from 0.025119, 50 to 2.584587, 60 fc rgb "#FFFF00"
set object 64 rect from 0.028196, 50 to 2.748633, 60 fc rgb "#FF00FF"
set object 65 rect from 0.029971, 50 to 2.822071, 60 fc rgb "#808080"
set object 66 rect from 0.030744, 50 to 2.895608, 60 fc rgb "#800080"
set object 67 rect from 0.031926, 50 to 2.998573, 60 fc rgb "#008080"
set object 68 rect from 0.032634, 50 to 144.159058, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
