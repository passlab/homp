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

set object 15 rect from 0.123091, 0 to 159.484666, 10 fc rgb "#FF0000"
set object 16 rect from 0.105058, 0 to 133.375556, 10 fc rgb "#00FF00"
set object 17 rect from 0.106532, 0 to 134.003330, 10 fc rgb "#0000FF"
set object 18 rect from 0.106831, 0 to 134.683837, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107388, 0 to 135.782441, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108248, 0 to 138.530834, 10 fc rgb "#808080"
set object 21 rect from 0.110439, 0 to 139.083276, 10 fc rgb "#800080"
set object 22 rect from 0.111105, 0 to 139.682171, 10 fc rgb "#008080"
set object 23 rect from 0.111354, 0 to 153.910036, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.121799, 10 to 158.152532, 20 fc rgb "#FF0000"
set object 25 rect from 0.092451, 10 to 117.705067, 20 fc rgb "#00FF00"
set object 26 rect from 0.094060, 10 to 118.425752, 20 fc rgb "#0000FF"
set object 27 rect from 0.094424, 10 to 119.169036, 20 fc rgb "#FFFF00"
set object 28 rect from 0.095048, 10 to 120.437140, 20 fc rgb "#FF00FF"
set object 29 rect from 0.096046, 10 to 123.180511, 20 fc rgb "#808080"
set object 30 rect from 0.098227, 10 to 123.798241, 20 fc rgb "#800080"
set object 31 rect from 0.098923, 10 to 124.451124, 20 fc rgb "#008080"
set object 32 rect from 0.099226, 10 to 152.270291, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.120386, 20 to 157.338937, 30 fc rgb "#FF0000"
set object 34 rect from 0.079733, 20 to 101.495950, 30 fc rgb "#00FF00"
set object 35 rect from 0.081164, 20 to 102.118701, 30 fc rgb "#0000FF"
set object 36 rect from 0.081435, 20 to 104.135111, 30 fc rgb "#FFFF00"
set object 37 rect from 0.083061, 20 to 105.661856, 30 fc rgb "#FF00FF"
set object 38 rect from 0.084261, 20 to 107.864086, 30 fc rgb "#808080"
set object 39 rect from 0.086013, 20 to 108.506927, 30 fc rgb "#800080"
set object 40 rect from 0.086735, 20 to 109.154790, 30 fc rgb "#008080"
set object 41 rect from 0.087041, 20 to 150.666957, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119159, 30 to 155.863667, 40 fc rgb "#FF0000"
set object 43 rect from 0.065669, 30 to 84.306248, 40 fc rgb "#00FF00"
set object 44 rect from 0.067455, 30 to 85.006844, 40 fc rgb "#0000FF"
set object 45 rect from 0.067807, 30 to 87.040831, 40 fc rgb "#FFFF00"
set object 46 rect from 0.069454, 30 to 88.366689, 40 fc rgb "#FF00FF"
set object 47 rect from 0.070490, 30 to 90.699497, 40 fc rgb "#808080"
set object 48 rect from 0.072339, 30 to 91.376236, 40 fc rgb "#800080"
set object 49 rect from 0.073107, 30 to 92.071809, 40 fc rgb "#008080"
set object 50 rect from 0.073434, 30 to 149.062368, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.117894, 40 to 154.454943, 50 fc rgb "#FF0000"
set object 52 rect from 0.051796, 40 to 67.209458, 50 fc rgb "#00FF00"
set object 53 rect from 0.053846, 40 to 67.892476, 50 fc rgb "#0000FF"
set object 54 rect from 0.054189, 40 to 70.113540, 50 fc rgb "#FFFF00"
set object 55 rect from 0.055953, 40 to 71.517242, 50 fc rgb "#FF00FF"
set object 56 rect from 0.057074, 40 to 73.755883, 50 fc rgb "#808080"
set object 57 rect from 0.058857, 40 to 74.481589, 50 fc rgb "#800080"
set object 58 rect from 0.059667, 40 to 75.229896, 50 fc rgb "#008080"
set object 59 rect from 0.060046, 40 to 147.361100, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116407, 50 to 157.117960, 60 fc rgb "#FF0000"
set object 61 rect from 0.015984, 50 to 39.427958, 60 fc rgb "#00FF00"
set object 62 rect from 0.031963, 50 to 41.071471, 60 fc rgb "#0000FF"
set object 63 rect from 0.032848, 50 to 46.234282, 60 fc rgb "#FFFF00"
set object 64 rect from 0.037028, 50 to 48.357412, 60 fc rgb "#FF00FF"
set object 65 rect from 0.038628, 50 to 50.789408, 60 fc rgb "#808080"
set object 66 rect from 0.040585, 50 to 51.688379, 60 fc rgb "#800080"
set object 67 rect from 0.041677, 50 to 53.000427, 60 fc rgb "#008080"
set object 68 rect from 0.042381, 50 to 145.070981, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
