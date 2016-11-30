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

set object 15 rect from 0.133191, 0 to 160.748353, 10 fc rgb "#FF0000"
set object 16 rect from 0.110924, 0 to 132.759865, 10 fc rgb "#00FF00"
set object 17 rect from 0.113097, 0 to 133.472233, 10 fc rgb "#0000FF"
set object 18 rect from 0.113471, 0 to 134.325901, 10 fc rgb "#FFFF00"
set object 19 rect from 0.114191, 0 to 135.488061, 10 fc rgb "#FF00FF"
set object 20 rect from 0.115190, 0 to 136.891611, 10 fc rgb "#808080"
set object 21 rect from 0.116377, 0 to 137.482699, 10 fc rgb "#800080"
set object 22 rect from 0.117129, 0 to 138.111470, 10 fc rgb "#008080"
set object 23 rect from 0.117412, 0 to 156.118534, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.131494, 10 to 159.353041, 20 fc rgb "#FF0000"
set object 25 rect from 0.095815, 10 to 115.042455, 20 fc rgb "#00FF00"
set object 26 rect from 0.098051, 10 to 115.851377, 20 fc rgb "#0000FF"
set object 27 rect from 0.098505, 10 to 116.902859, 20 fc rgb "#FFFF00"
set object 28 rect from 0.099427, 10 to 118.251065, 20 fc rgb "#FF00FF"
set object 29 rect from 0.100575, 10 to 119.727613, 20 fc rgb "#808080"
set object 30 rect from 0.101829, 10 to 120.491790, 20 fc rgb "#800080"
set object 31 rect from 0.102697, 10 to 121.177080, 20 fc rgb "#008080"
set object 32 rect from 0.103029, 10 to 154.105056, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129709, 20 to 157.052267, 30 fc rgb "#FF0000"
set object 34 rect from 0.081470, 20 to 97.879637, 30 fc rgb "#00FF00"
set object 35 rect from 0.083491, 20 to 98.650880, 30 fc rgb "#0000FF"
set object 36 rect from 0.083892, 20 to 99.662327, 30 fc rgb "#FFFF00"
set object 37 rect from 0.084755, 20 to 100.931644, 30 fc rgb "#FF00FF"
set object 38 rect from 0.085831, 20 to 102.284556, 30 fc rgb "#808080"
set object 39 rect from 0.086985, 20 to 103.033428, 30 fc rgb "#800080"
set object 40 rect from 0.087864, 20 to 103.732847, 30 fc rgb "#008080"
set object 41 rect from 0.088238, 20 to 152.092759, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127896, 30 to 155.023477, 40 fc rgb "#FF0000"
set object 43 rect from 0.064740, 30 to 78.225288, 40 fc rgb "#00FF00"
set object 44 rect from 0.066887, 30 to 79.038921, 40 fc rgb "#0000FF"
set object 45 rect from 0.067260, 30 to 80.195200, 40 fc rgb "#FFFF00"
set object 46 rect from 0.068222, 30 to 81.537515, 40 fc rgb "#FF00FF"
set object 47 rect from 0.069380, 30 to 82.924576, 40 fc rgb "#808080"
set object 48 rect from 0.070558, 30 to 83.645190, 40 fc rgb "#800080"
set object 49 rect from 0.071415, 30 to 84.378754, 40 fc rgb "#008080"
set object 50 rect from 0.071789, 30 to 149.900307, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126212, 40 to 153.234904, 50 fc rgb "#FF0000"
set object 52 rect from 0.048632, 40 to 59.757835, 50 fc rgb "#00FF00"
set object 53 rect from 0.051107, 40 to 60.470203, 50 fc rgb "#0000FF"
set object 54 rect from 0.051466, 40 to 61.520510, 50 fc rgb "#FFFF00"
set object 55 rect from 0.052383, 40 to 62.937004, 50 fc rgb "#FF00FF"
set object 56 rect from 0.053578, 40 to 64.471251, 50 fc rgb "#808080"
set object 57 rect from 0.054878, 40 to 65.240138, 50 fc rgb "#800080"
set object 58 rect from 0.055789, 40 to 65.990190, 50 fc rgb "#008080"
set object 59 rect from 0.056202, 40 to 147.838558, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.124337, 50 to 154.014389, 60 fc rgb "#FF0000"
set object 61 rect from 0.026985, 50 to 35.216982, 60 fc rgb "#00FF00"
set object 62 rect from 0.030347, 50 to 36.740628, 60 fc rgb "#0000FF"
set object 63 rect from 0.031336, 50 to 39.166218, 60 fc rgb "#FFFF00"
set object 64 rect from 0.033414, 50 to 41.090208, 60 fc rgb "#FF00FF"
set object 65 rect from 0.035032, 50 to 42.904690, 60 fc rgb "#808080"
set object 66 rect from 0.036594, 50 to 43.860799, 60 fc rgb "#800080"
set object 67 rect from 0.037806, 50 to 45.157192, 60 fc rgb "#008080"
set object 68 rect from 0.038491, 50 to 145.304640, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
