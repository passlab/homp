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

set object 15 rect from 0.109187, 0 to 162.032004, 10 fc rgb "#FF0000"
set object 16 rect from 0.090002, 0 to 132.959442, 10 fc rgb "#00FF00"
set object 17 rect from 0.092047, 0 to 133.844069, 10 fc rgb "#0000FF"
set object 18 rect from 0.092413, 0 to 134.844700, 10 fc rgb "#FFFF00"
set object 19 rect from 0.093126, 0 to 136.350053, 10 fc rgb "#FF00FF"
set object 20 rect from 0.094139, 0 to 136.978041, 10 fc rgb "#808080"
set object 21 rect from 0.094576, 0 to 137.734996, 10 fc rgb "#800080"
set object 22 rect from 0.095405, 0 to 138.563092, 10 fc rgb "#008080"
set object 23 rect from 0.095670, 0 to 157.555273, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.107675, 10 to 160.345559, 20 fc rgb "#FF0000"
set object 25 rect from 0.076135, 10 to 112.736095, 20 fc rgb "#00FF00"
set object 26 rect from 0.078107, 10 to 113.774413, 20 fc rgb "#0000FF"
set object 27 rect from 0.078585, 10 to 114.830192, 20 fc rgb "#FFFF00"
set object 28 rect from 0.079345, 10 to 116.598064, 20 fc rgb "#FF00FF"
set object 29 rect from 0.080549, 10 to 117.260801, 20 fc rgb "#808080"
set object 30 rect from 0.081021, 10 to 118.175855, 20 fc rgb "#800080"
set object 31 rect from 0.081910, 10 to 119.101110, 20 fc rgb "#008080"
set object 32 rect from 0.082250, 10 to 155.220354, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.106034, 20 to 157.484046, 30 fc rgb "#FF0000"
set object 34 rect from 0.063734, 20 to 94.705473, 30 fc rgb "#00FF00"
set object 35 rect from 0.065676, 20 to 95.542213, 30 fc rgb "#0000FF"
set object 36 rect from 0.066002, 20 to 96.619775, 30 fc rgb "#FFFF00"
set object 37 rect from 0.066758, 20 to 98.143887, 30 fc rgb "#FF00FF"
set object 38 rect from 0.067796, 20 to 98.718195, 30 fc rgb "#808080"
set object 39 rect from 0.068194, 20 to 99.505664, 30 fc rgb "#800080"
set object 40 rect from 0.069019, 20 to 100.455554, 30 fc rgb "#008080"
set object 41 rect from 0.069390, 20 to 153.018984, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.104473, 30 to 155.543813, 40 fc rgb "#FF0000"
set object 43 rect from 0.050544, 30 to 75.536350, 40 fc rgb "#00FF00"
set object 44 rect from 0.052464, 30 to 76.451491, 40 fc rgb "#0000FF"
set object 45 rect from 0.052838, 30 to 77.649377, 40 fc rgb "#FFFF00"
set object 46 rect from 0.053666, 30 to 79.264942, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054776, 30 to 79.846425, 40 fc rgb "#808080"
set object 48 rect from 0.055183, 30 to 80.649885, 40 fc rgb "#800080"
set object 49 rect from 0.056031, 30 to 81.575140, 40 fc rgb "#008080"
set object 50 rect from 0.056371, 30 to 150.718900, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.102950, 40 to 153.214685, 50 fc rgb "#FF0000"
set object 52 rect from 0.037305, 40 to 57.057623, 50 fc rgb "#00FF00"
set object 53 rect from 0.039710, 40 to 57.964034, 50 fc rgb "#0000FF"
set object 54 rect from 0.040090, 40 to 59.053179, 50 fc rgb "#FFFF00"
set object 55 rect from 0.040841, 40 to 60.697701, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041995, 40 to 61.257487, 50 fc rgb "#808080"
set object 57 rect from 0.042375, 40 to 62.078322, 50 fc rgb "#800080"
set object 58 rect from 0.043228, 40 to 63.038325, 50 fc rgb "#008080"
set object 59 rect from 0.043604, 40 to 148.379659, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.101222, 50 to 154.565052, 60 fc rgb "#FF0000"
set object 61 rect from 0.017965, 50 to 29.851474, 60 fc rgb "#00FF00"
set object 62 rect from 0.021057, 50 to 31.471361, 60 fc rgb "#0000FF"
set object 63 rect from 0.021838, 50 to 34.521140, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023996, 50 to 36.973531, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025631, 50 to 37.879856, 60 fc rgb "#808080"
set object 66 rect from 0.026264, 50 to 38.989314, 60 fc rgb "#800080"
set object 67 rect from 0.027653, 50 to 40.894885, 60 fc rgb "#008080"
set object 68 rect from 0.028339, 50 to 145.383560, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
