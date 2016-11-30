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

set object 15 rect from 0.124375, 0 to 160.535573, 10 fc rgb "#FF0000"
set object 16 rect from 0.103683, 0 to 133.446115, 10 fc rgb "#00FF00"
set object 17 rect from 0.105915, 0 to 134.256457, 10 fc rgb "#0000FF"
set object 18 rect from 0.106320, 0 to 135.157741, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107069, 0 to 136.512832, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108106, 0 to 137.041230, 10 fc rgb "#808080"
set object 21 rect from 0.108546, 0 to 137.699826, 10 fc rgb "#800080"
set object 22 rect from 0.109322, 0 to 138.384944, 10 fc rgb "#008080"
set object 23 rect from 0.109610, 0 to 156.549941, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.122785, 10 to 159.060457, 20 fc rgb "#FF0000"
set object 25 rect from 0.088585, 10 to 114.338167, 20 fc rgb "#00FF00"
set object 26 rect from 0.090917, 10 to 115.372210, 20 fc rgb "#0000FF"
set object 27 rect from 0.091386, 10 to 116.372121, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092213, 10 to 117.948427, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093467, 10 to 118.593084, 20 fc rgb "#808080"
set object 30 rect from 0.093954, 10 to 119.341417, 20 fc rgb "#800080"
set object 31 rect from 0.094778, 10 to 120.075886, 20 fc rgb "#008080"
set object 32 rect from 0.095111, 10 to 154.369364, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121034, 20 to 156.413415, 30 fc rgb "#FF0000"
set object 34 rect from 0.074020, 20 to 95.917823, 30 fc rgb "#00FF00"
set object 35 rect from 0.076241, 20 to 96.666231, 30 fc rgb "#0000FF"
set object 36 rect from 0.076579, 20 to 97.693870, 30 fc rgb "#FFFF00"
set object 37 rect from 0.077399, 20 to 99.065461, 30 fc rgb "#FF00FF"
set object 38 rect from 0.078492, 20 to 99.593860, 30 fc rgb "#808080"
set object 39 rect from 0.078925, 20 to 100.258710, 30 fc rgb "#800080"
set object 40 rect from 0.079695, 20 to 100.988130, 30 fc rgb "#008080"
set object 41 rect from 0.080008, 20 to 152.397343, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119505, 30 to 154.680315, 40 fc rgb "#FF0000"
set object 43 rect from 0.059209, 30 to 77.161287, 40 fc rgb "#00FF00"
set object 44 rect from 0.061497, 30 to 78.028440, 40 fc rgb "#0000FF"
set object 45 rect from 0.061840, 30 to 79.197728, 40 fc rgb "#FFFF00"
set object 46 rect from 0.062765, 30 to 80.594561, 40 fc rgb "#FF00FF"
set object 47 rect from 0.063878, 30 to 81.129288, 40 fc rgb "#808080"
set object 48 rect from 0.064299, 30 to 81.796700, 40 fc rgb "#800080"
set object 49 rect from 0.065099, 30 to 82.537498, 40 fc rgb "#008080"
set object 50 rect from 0.065426, 30 to 150.382451, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.117780, 40 to 152.837286, 50 fc rgb "#FF0000"
set object 52 rect from 0.044258, 40 to 58.620918, 50 fc rgb "#00FF00"
set object 53 rect from 0.046757, 40 to 59.489351, 50 fc rgb "#0000FF"
set object 54 rect from 0.047191, 40 to 60.744534, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048199, 40 to 62.228616, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049371, 40 to 62.902357, 50 fc rgb "#808080"
set object 57 rect from 0.049905, 40 to 63.627935, 50 fc rgb "#800080"
set object 58 rect from 0.050764, 40 to 64.464799, 50 fc rgb "#008080"
set object 59 rect from 0.051120, 40 to 148.074163, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.115962, 50 to 153.900413, 60 fc rgb "#FF0000"
set object 61 rect from 0.023932, 50 to 33.808102, 60 fc rgb "#00FF00"
set object 62 rect from 0.027181, 50 to 35.335132, 60 fc rgb "#0000FF"
set object 63 rect from 0.028076, 50 to 38.056690, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030300, 50 to 40.526745, 60 fc rgb "#FF00FF"
set object 65 rect from 0.032196, 50 to 41.379959, 60 fc rgb "#808080"
set object 66 rect from 0.032879, 50 to 42.269866, 60 fc rgb "#800080"
set object 67 rect from 0.033973, 50 to 43.604764, 60 fc rgb "#008080"
set object 68 rect from 0.034634, 50 to 145.360214, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
