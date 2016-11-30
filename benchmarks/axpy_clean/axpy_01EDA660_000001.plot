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

set object 15 rect from 0.999282, 0 to 154.898044, 10 fc rgb "#FF0000"
set object 16 rect from 0.167754, 0 to 27.617642, 10 fc rgb "#00FF00"
set object 17 rect from 0.189161, 0 to 28.600076, 10 fc rgb "#0000FF"
set object 18 rect from 0.193833, 0 to 29.911024, 10 fc rgb "#FFFF00"
set object 19 rect from 0.202869, 0 to 31.333850, 10 fc rgb "#FF00FF"
set object 20 rect from 0.212381, 0 to 34.601070, 10 fc rgb "#808080"
set object 21 rect from 0.234407, 0 to 35.257803, 10 fc rgb "#800080"
set object 22 rect from 0.241293, 0 to 35.986406, 10 fc rgb "#008080"
set object 23 rect from 0.243721, 0 to 147.369464, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.023228, 10 to 156.894771, 20 fc rgb "#FF0000"
set object 25 rect from 0.529791, 10 to 79.977937, 20 fc rgb "#00FF00"
set object 26 rect from 0.541999, 10 to 80.570954, 20 fc rgb "#0000FF"
set object 27 rect from 0.544493, 10 to 81.191828, 20 fc rgb "#FFFF00"
set object 28 rect from 0.548761, 10 to 82.242277, 20 fc rgb "#FF00FF"
set object 29 rect from 0.555841, 10 to 85.378063, 20 fc rgb "#808080"
set object 30 rect from 0.577121, 10 to 85.972559, 20 fc rgb "#800080"
set object 31 rect from 0.582691, 10 to 86.535200, 20 fc rgb "#008080"
set object 32 rect from 0.584803, 10 to 150.956605, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.034164, 20 to 159.860893, 30 fc rgb "#FF0000"
set object 34 rect from 0.624952, 20 to 93.959462, 30 fc rgb "#00FF00"
set object 35 rect from 0.636380, 20 to 94.508914, 30 fc rgb "#0000FF"
set object 36 rect from 0.638559, 20 to 96.303669, 30 fc rgb "#FFFF00"
set object 37 rect from 0.650871, 20 to 97.723976, 30 fc rgb "#FF00FF"
set object 38 rect from 0.660347, 20 to 100.716174, 30 fc rgb "#808080"
set object 39 rect from 0.680544, 20 to 101.313784, 30 fc rgb "#800080"
set object 40 rect from 0.686239, 20 to 101.925026, 30 fc rgb "#008080"
set object 41 rect from 0.688603, 20 to 152.705872, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.043776, 30 to 161.553554, 40 fc rgb "#FF0000"
set object 43 rect from 0.867570, 30 to 130.063965, 40 fc rgb "#00FF00"
set object 44 rect from 0.880343, 30 to 130.753448, 40 fc rgb "#0000FF"
set object 45 rect from 0.883144, 30 to 132.630592, 40 fc rgb "#FFFF00"
set object 46 rect from 0.895914, 30 to 134.051934, 40 fc rgb "#FF00FF"
set object 47 rect from 0.905400, 30 to 137.088143, 40 fc rgb "#808080"
set object 48 rect from 0.925924, 30 to 137.687679, 40 fc rgb "#800080"
set object 49 rect from 0.931588, 30 to 138.398646, 40 fc rgb "#008080"
set object 50 rect from 0.934759, 30 to 154.227828, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.986001, 40 to 153.459368, 50 fc rgb "#FF0000"
set object 52 rect from 0.297921, 40 to 45.971366, 50 fc rgb "#00FF00"
set object 53 rect from 0.312524, 40 to 46.693596, 50 fc rgb "#0000FF"
set object 54 rect from 0.315861, 40 to 48.626456, 50 fc rgb "#FFFF00"
set object 55 rect from 0.329149, 40 to 50.337934, 50 fc rgb "#FF00FF"
set object 56 rect from 0.340644, 40 to 53.363178, 50 fc rgb "#808080"
set object 57 rect from 0.361064, 40 to 54.069404, 50 fc rgb "#800080"
set object 58 rect from 0.367688, 40 to 55.101478, 50 fc rgb "#008080"
set object 59 rect from 0.372732, 40 to 145.215964, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.011442, 50 to 156.894180, 60 fc rgb "#FF0000"
set object 61 rect from 0.414857, 50 to 62.924196, 60 fc rgb "#00FF00"
set object 62 rect from 0.426972, 50 to 63.487725, 60 fc rgb "#0000FF"
set object 63 rect from 0.429202, 50 to 65.685975, 60 fc rgb "#FFFF00"
set object 64 rect from 0.444083, 50 to 67.071163, 60 fc rgb "#FF00FF"
set object 65 rect from 0.453381, 50 to 70.043060, 60 fc rgb "#808080"
set object 66 rect from 0.473452, 50 to 70.627927, 60 fc rgb "#800080"
set object 67 rect from 0.479160, 50 to 71.273547, 60 fc rgb "#008080"
set object 68 rect from 0.481753, 50 to 149.228531, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
