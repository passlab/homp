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

set object 15 rect from 0.146834, 0 to 154.398608, 10 fc rgb "#FF0000"
set object 16 rect from 0.078625, 0 to 82.830133, 10 fc rgb "#00FF00"
set object 17 rect from 0.081198, 0 to 83.601957, 10 fc rgb "#0000FF"
set object 18 rect from 0.081691, 0 to 84.463980, 10 fc rgb "#FFFF00"
set object 19 rect from 0.082532, 0 to 85.581230, 10 fc rgb "#FF00FF"
set object 20 rect from 0.083633, 0 to 86.994705, 10 fc rgb "#808080"
set object 21 rect from 0.085024, 0 to 87.567677, 10 fc rgb "#800080"
set object 22 rect from 0.085833, 0 to 88.169353, 10 fc rgb "#008080"
set object 23 rect from 0.086134, 0 to 149.726664, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.142742, 10 to 153.779505, 20 fc rgb "#FF0000"
set object 25 rect from 0.038097, 10 to 42.851091, 20 fc rgb "#00FF00"
set object 26 rect from 0.042263, 10 to 44.158987, 20 fc rgb "#0000FF"
set object 27 rect from 0.043206, 10 to 46.361709, 20 fc rgb "#FFFF00"
set object 28 rect from 0.045402, 10 to 48.261032, 20 fc rgb "#FF00FF"
set object 29 rect from 0.047214, 10 to 50.283354, 20 fc rgb "#808080"
set object 30 rect from 0.049199, 10 to 51.231478, 20 fc rgb "#800080"
set object 31 rect from 0.050604, 10 to 52.120154, 20 fc rgb "#008080"
set object 32 rect from 0.051137, 10 to 145.117246, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.148621, 20 to 156.847332, 30 fc rgb "#FF0000"
set object 34 rect from 0.095454, 20 to 99.873810, 30 fc rgb "#00FF00"
set object 35 rect from 0.097818, 20 to 100.868061, 30 fc rgb "#0000FF"
set object 36 rect from 0.098522, 20 to 101.798758, 30 fc rgb "#FFFF00"
set object 37 rect from 0.099447, 20 to 103.149706, 30 fc rgb "#FF00FF"
set object 38 rect from 0.100774, 20 to 104.433004, 30 fc rgb "#808080"
set object 39 rect from 0.102028, 20 to 105.197653, 30 fc rgb "#800080"
set object 40 rect from 0.103038, 20 to 106.007402, 30 fc rgb "#008080"
set object 41 rect from 0.103546, 20 to 151.687486, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.144854, 30 to 153.633958, 40 fc rgb "#FF0000"
set object 43 rect from 0.061232, 30 to 65.038210, 40 fc rgb "#00FF00"
set object 44 rect from 0.063824, 30 to 66.045784, 40 fc rgb "#0000FF"
set object 45 rect from 0.064568, 30 to 67.241956, 40 fc rgb "#FFFF00"
set object 46 rect from 0.065746, 30 to 68.948579, 40 fc rgb "#FF00FF"
set object 47 rect from 0.067387, 30 to 70.217527, 40 fc rgb "#808080"
set object 48 rect from 0.068649, 30 to 71.009852, 40 fc rgb "#800080"
set object 49 rect from 0.069678, 30 to 72.081999, 40 fc rgb "#008080"
set object 50 rect from 0.070488, 30 to 147.654118, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.151756, 40 to 159.366777, 50 fc rgb "#FF0000"
set object 52 rect from 0.127786, 40 to 132.818288, 50 fc rgb "#00FF00"
set object 53 rect from 0.129940, 40 to 133.545013, 50 fc rgb "#0000FF"
set object 54 rect from 0.130415, 40 to 134.426510, 50 fc rgb "#FFFF00"
set object 55 rect from 0.131264, 40 to 135.555036, 50 fc rgb "#FF00FF"
set object 56 rect from 0.132363, 40 to 136.747107, 50 fc rgb "#808080"
set object 57 rect from 0.133530, 40 to 137.376459, 50 fc rgb "#800080"
set object 58 rect from 0.134399, 40 to 138.031432, 50 fc rgb "#008080"
set object 59 rect from 0.134781, 40 to 154.967483, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.150166, 50 to 157.978931, 60 fc rgb "#FF0000"
set object 61 rect from 0.110991, 50 to 115.436365, 60 fc rgb "#00FF00"
set object 62 rect from 0.113001, 50 to 116.274813, 60 fc rgb "#0000FF"
set object 63 rect from 0.113553, 50 to 117.203459, 60 fc rgb "#FFFF00"
set object 64 rect from 0.114468, 50 to 118.382209, 60 fc rgb "#FF00FF"
set object 65 rect from 0.115607, 50 to 119.586583, 60 fc rgb "#808080"
set object 66 rect from 0.116802, 50 to 120.233358, 60 fc rgb "#800080"
set object 67 rect from 0.117680, 50 to 120.880129, 60 fc rgb "#008080"
set object 68 rect from 0.118049, 50 to 153.432036, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
