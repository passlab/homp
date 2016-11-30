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

set object 15 rect from 2.126997, 0 to 145.340624, 10 fc rgb "#FF0000"
set object 16 rect from 2.096212, 0 to 143.114935, 10 fc rgb "#00FF00"
set object 17 rect from 2.098939, 0 to 143.173722, 10 fc rgb "#0000FF"
set object 18 rect from 2.099546, 0 to 143.233053, 10 fc rgb "#FFFF00"
set object 19 rect from 2.100394, 0 to 143.308548, 10 fc rgb "#FF00FF"
set object 20 rect from 2.101508, 0 to 143.414527, 10 fc rgb "#808080"
set object 21 rect from 2.103062, 0 to 143.454558, 10 fc rgb "#800080"
set object 22 rect from 2.103936, 0 to 143.493977, 10 fc rgb "#008080"
set object 23 rect from 2.104235, 0 to 145.017369, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 2.125406, 10 to 145.331895, 20 fc rgb "#FF0000"
set object 25 rect from 2.060091, 10 to 140.681925, 20 fc rgb "#00FF00"
set object 26 rect from 2.063354, 10 to 140.764036, 20 fc rgb "#0000FF"
set object 27 rect from 2.064217, 10 to 140.866331, 20 fc rgb "#FFFF00"
set object 28 rect from 2.065721, 10 to 140.960920, 20 fc rgb "#FF00FF"
set object 29 rect from 2.067095, 10 to 141.083609, 20 fc rgb "#808080"
set object 30 rect from 2.068901, 10 to 141.132027, 20 fc rgb "#800080"
set object 31 rect from 2.069979, 10 to 141.182698, 20 fc rgb "#008080"
set object 32 rect from 2.070346, 10 to 144.901706, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 2.123463, 20 to 145.081545, 30 fc rgb "#FF0000"
set object 34 rect from 0.074011, 20 to 5.176046, 30 fc rgb "#00FF00"
set object 35 rect from 0.076301, 20 to 5.223921, 30 fc rgb "#0000FF"
set object 36 rect from 0.076723, 20 to 5.282024, 30 fc rgb "#FFFF00"
set object 37 rect from 0.077583, 20 to 5.366454, 30 fc rgb "#FF00FF"
set object 38 rect from 0.078810, 20 to 5.451767, 30 fc rgb "#808080"
set object 39 rect from 0.080067, 20 to 5.496369, 30 fc rgb "#800080"
set object 40 rect from 0.080990, 20 to 5.539741, 30 fc rgb "#008080"
set object 41 rect from 2.041849, 20 to 144.755968, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 2.121156, 30 to 144.928848, 40 fc rgb "#FF0000"
set object 43 rect from 0.057811, 30 to 4.064224, 40 fc rgb "#00FF00"
set object 44 rect from 0.060071, 30 to 4.117145, 40 fc rgb "#0000FF"
set object 45 rect from 0.060490, 30 to 4.182478, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061461, 30 to 4.262678, 40 fc rgb "#FF00FF"
set object 47 rect from 0.062627, 30 to 4.346288, 40 fc rgb "#808080"
set object 48 rect from 0.063859, 30 to 4.394163, 40 fc rgb "#800080"
set object 49 rect from 0.064829, 30 to 4.439445, 40 fc rgb "#008080"
set object 50 rect from 0.065226, 30 to 144.574565, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 2.118061, 40 to 144.722961, 50 fc rgb "#FF0000"
set object 52 rect from 0.041862, 40 to 3.006006, 50 fc rgb "#00FF00"
set object 53 rect from 0.044580, 40 to 3.060768, 50 fc rgb "#0000FF"
set object 54 rect from 0.045004, 40 to 3.121463, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045900, 40 to 3.212780, 50 fc rgb "#FF00FF"
set object 56 rect from 0.047244, 40 to 3.299185, 50 fc rgb "#808080"
set object 57 rect from 0.048540, 40 to 3.346173, 50 fc rgb "#800080"
set object 58 rect from 0.049465, 40 to 3.393980, 50 fc rgb "#008080"
set object 59 rect from 0.049914, 40 to 144.369768, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 2.114847, 50 to 144.682520, 60 fc rgb "#FF0000"
set object 61 rect from 0.019617, 50 to 1.548968, 60 fc rgb "#00FF00"
set object 62 rect from 0.023581, 50 to 1.655492, 60 fc rgb "#0000FF"
set object 63 rect from 0.024413, 50 to 1.795364, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026530, 50 to 1.933328, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028487, 50 to 2.037397, 60 fc rgb "#808080"
set object 66 rect from 0.030030, 50 to 2.096592, 60 fc rgb "#800080"
set object 67 rect from 0.031407, 50 to 2.189205, 60 fc rgb "#008080"
set object 68 rect from 0.032235, 50 to 144.122824, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
