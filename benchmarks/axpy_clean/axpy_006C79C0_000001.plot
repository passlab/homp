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

set object 15 rect from 0.119778, 0 to 161.164711, 10 fc rgb "#FF0000"
set object 16 rect from 0.099746, 0 to 132.682307, 10 fc rgb "#00FF00"
set object 17 rect from 0.101565, 0 to 133.538574, 10 fc rgb "#0000FF"
set object 18 rect from 0.101959, 0 to 134.384358, 10 fc rgb "#FFFF00"
set object 19 rect from 0.102603, 0 to 135.708746, 10 fc rgb "#FF00FF"
set object 20 rect from 0.103613, 0 to 137.085598, 10 fc rgb "#808080"
set object 21 rect from 0.104666, 0 to 137.758290, 10 fc rgb "#800080"
set object 22 rect from 0.105455, 0 to 138.476871, 10 fc rgb "#008080"
set object 23 rect from 0.105748, 0 to 156.383767, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.118311, 10 to 159.772120, 20 fc rgb "#FF0000"
set object 25 rect from 0.085701, 10 to 114.544618, 20 fc rgb "#00FF00"
set object 26 rect from 0.087727, 10 to 115.420562, 20 fc rgb "#0000FF"
set object 27 rect from 0.088142, 10 to 116.393529, 20 fc rgb "#FFFF00"
set object 28 rect from 0.088925, 10 to 117.995918, 20 fc rgb "#FF00FF"
set object 29 rect from 0.090138, 10 to 119.464558, 20 fc rgb "#808080"
set object 30 rect from 0.091252, 10 to 120.274936, 20 fc rgb "#800080"
set object 31 rect from 0.092106, 10 to 121.069575, 20 fc rgb "#008080"
set object 32 rect from 0.092455, 10 to 154.256865, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.116591, 20 to 157.069570, 30 fc rgb "#FF0000"
set object 34 rect from 0.072523, 20 to 96.869824, 30 fc rgb "#00FF00"
set object 35 rect from 0.074270, 20 to 97.643477, 30 fc rgb "#0000FF"
set object 36 rect from 0.074585, 20 to 98.685948, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075379, 20 to 100.129674, 30 fc rgb "#FF00FF"
set object 38 rect from 0.076478, 20 to 101.299338, 30 fc rgb "#808080"
set object 39 rect from 0.077391, 20 to 102.041523, 30 fc rgb "#800080"
set object 40 rect from 0.078217, 20 to 102.921393, 30 fc rgb "#008080"
set object 41 rect from 0.078613, 20 to 152.280761, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.115163, 30 to 155.488157, 40 fc rgb "#FF0000"
set object 43 rect from 0.058530, 30 to 78.725580, 40 fc rgb "#00FF00"
set object 44 rect from 0.060426, 30 to 79.533330, 40 fc rgb "#0000FF"
set object 45 rect from 0.060776, 30 to 80.706931, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061668, 30 to 82.250310, 40 fc rgb "#FF00FF"
set object 47 rect from 0.062845, 30 to 83.442268, 40 fc rgb "#808080"
set object 48 rect from 0.063767, 30 to 84.209366, 40 fc rgb "#800080"
set object 49 rect from 0.064604, 30 to 85.044657, 40 fc rgb "#008080"
set object 50 rect from 0.064978, 30 to 150.368913, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.111896, 40 to 151.328779, 50 fc rgb "#FF0000"
set object 52 rect from 0.044417, 40 to 60.656085, 50 fc rgb "#00FF00"
set object 53 rect from 0.046654, 40 to 61.491366, 50 fc rgb "#0000FF"
set object 54 rect from 0.047014, 40 to 62.604648, 50 fc rgb "#FFFF00"
set object 55 rect from 0.047865, 40 to 64.199172, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049105, 40 to 65.498657, 50 fc rgb "#808080"
set object 57 rect from 0.050068, 40 to 66.263137, 50 fc rgb "#800080"
set object 58 rect from 0.050914, 40 to 67.141688, 50 fc rgb "#008080"
set object 59 rect from 0.051348, 40 to 145.350620, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.113698, 50 to 156.688000, 60 fc rgb "#FF0000"
set object 61 rect from 0.025035, 50 to 36.344870, 60 fc rgb "#00FF00"
set object 62 rect from 0.028188, 50 to 37.850225, 60 fc rgb "#0000FF"
set object 63 rect from 0.028989, 50 to 40.479345, 60 fc rgb "#FFFF00"
set object 64 rect from 0.031034, 50 to 42.594454, 60 fc rgb "#FF00FF"
set object 65 rect from 0.032619, 50 to 44.160128, 60 fc rgb "#808080"
set object 66 rect from 0.033839, 50 to 45.138342, 60 fc rgb "#800080"
set object 67 rect from 0.035070, 50 to 46.764335, 60 fc rgb "#008080"
set object 68 rect from 0.035793, 50 to 148.273480, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
