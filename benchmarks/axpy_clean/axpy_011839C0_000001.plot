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

set object 15 rect from 0.096457, 0 to 162.404513, 10 fc rgb "#FF0000"
set object 16 rect from 0.079440, 0 to 132.136824, 10 fc rgb "#00FF00"
set object 17 rect from 0.081033, 0 to 133.055166, 10 fc rgb "#0000FF"
set object 18 rect from 0.081404, 0 to 134.029173, 10 fc rgb "#FFFF00"
set object 19 rect from 0.081991, 0 to 135.428787, 10 fc rgb "#FF00FF"
set object 20 rect from 0.082833, 0 to 136.939730, 10 fc rgb "#808080"
set object 21 rect from 0.083780, 0 to 137.756598, 10 fc rgb "#800080"
set object 22 rect from 0.084476, 0 to 138.476870, 10 fc rgb "#008080"
set object 23 rect from 0.084695, 0 to 157.154778, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.095186, 10 to 160.682426, 20 fc rgb "#FF0000"
set object 25 rect from 0.066960, 10 to 111.745024, 20 fc rgb "#00FF00"
set object 26 rect from 0.068571, 10 to 112.745180, 20 fc rgb "#0000FF"
set object 27 rect from 0.068974, 10 to 113.711040, 20 fc rgb "#FFFF00"
set object 28 rect from 0.069593, 10 to 115.357802, 20 fc rgb "#FF00FF"
set object 29 rect from 0.070596, 10 to 116.852353, 20 fc rgb "#808080"
set object 30 rect from 0.071514, 10 to 117.742888, 20 fc rgb "#800080"
set object 31 rect from 0.072294, 10 to 118.626885, 20 fc rgb "#008080"
set object 32 rect from 0.072572, 10 to 154.926832, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.093678, 20 to 157.950376, 30 fc rgb "#FF0000"
set object 34 rect from 0.056214, 20 to 93.923257, 30 fc rgb "#00FF00"
set object 35 rect from 0.057713, 20 to 94.849746, 30 fc rgb "#0000FF"
set object 36 rect from 0.058043, 20 to 95.935082, 30 fc rgb "#FFFF00"
set object 37 rect from 0.058719, 20 to 97.668634, 30 fc rgb "#FF00FF"
set object 38 rect from 0.059762, 20 to 98.855445, 30 fc rgb "#808080"
set object 39 rect from 0.060489, 20 to 99.592109, 30 fc rgb "#800080"
set object 40 rect from 0.061175, 20 to 100.462983, 30 fc rgb "#008080"
set object 41 rect from 0.061491, 20 to 152.769283, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.092468, 30 to 156.151256, 40 fc rgb "#FF0000"
set object 43 rect from 0.044235, 30 to 74.580692, 40 fc rgb "#00FF00"
set object 44 rect from 0.045867, 30 to 75.589093, 40 fc rgb "#0000FF"
set object 45 rect from 0.046277, 30 to 76.667843, 40 fc rgb "#FFFF00"
set object 46 rect from 0.046942, 30 to 78.331046, 40 fc rgb "#FF00FF"
set object 47 rect from 0.047950, 30 to 79.578399, 40 fc rgb "#808080"
set object 48 rect from 0.048722, 30 to 80.387071, 40 fc rgb "#800080"
set object 49 rect from 0.049421, 30 to 81.243212, 40 fc rgb "#008080"
set object 50 rect from 0.049731, 30 to 150.608466, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.091148, 40 to 154.095279, 50 fc rgb "#FF0000"
set object 52 rect from 0.032232, 40 to 55.388777, 50 fc rgb "#00FF00"
set object 53 rect from 0.034169, 40 to 56.469186, 50 fc rgb "#0000FF"
set object 54 rect from 0.034601, 40 to 57.551253, 50 fc rgb "#FFFF00"
set object 55 rect from 0.035264, 40 to 59.302807, 50 fc rgb "#FF00FF"
set object 56 rect from 0.036340, 40 to 60.589482, 50 fc rgb "#808080"
set object 57 rect from 0.037118, 40 to 61.357223, 50 fc rgb "#800080"
set object 58 rect from 0.037817, 40 to 62.296836, 50 fc rgb "#008080"
set object 59 rect from 0.038170, 40 to 148.341295, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.089670, 50 to 154.583039, 60 fc rgb "#FF0000"
set object 61 rect from 0.015873, 50 to 29.851937, 60 fc rgb "#00FF00"
set object 62 rect from 0.018637, 50 to 31.395615, 60 fc rgb "#0000FF"
set object 63 rect from 0.019291, 50 to 34.106443, 60 fc rgb "#FFFF00"
set object 64 rect from 0.020974, 50 to 36.471868, 60 fc rgb "#FF00FF"
set object 65 rect from 0.022396, 50 to 37.925536, 60 fc rgb "#808080"
set object 66 rect from 0.023287, 50 to 38.919155, 60 fc rgb "#800080"
set object 67 rect from 0.024269, 50 to 40.577431, 60 fc rgb "#008080"
set object 68 rect from 0.024904, 50 to 145.424202, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
