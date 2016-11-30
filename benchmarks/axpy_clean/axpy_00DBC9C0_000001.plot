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

set object 15 rect from 0.123192, 0 to 162.066707, 10 fc rgb "#FF0000"
set object 16 rect from 0.101351, 0 to 132.109530, 10 fc rgb "#00FF00"
set object 17 rect from 0.103601, 0 to 132.977222, 10 fc rgb "#0000FF"
set object 18 rect from 0.104032, 0 to 133.976733, 10 fc rgb "#FFFF00"
set object 19 rect from 0.104802, 0 to 135.255237, 10 fc rgb "#FF00FF"
set object 20 rect from 0.105804, 0 to 136.825532, 10 fc rgb "#808080"
set object 21 rect from 0.107029, 0 to 137.455187, 10 fc rgb "#800080"
set object 22 rect from 0.107777, 0 to 138.110435, 10 fc rgb "#008080"
set object 23 rect from 0.108035, 0 to 156.923259, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.121598, 10 to 160.394033, 20 fc rgb "#FF0000"
set object 25 rect from 0.087010, 10 to 113.711357, 20 fc rgb "#00FF00"
set object 26 rect from 0.089197, 10 to 114.531698, 20 fc rgb "#0000FF"
set object 27 rect from 0.089603, 10 to 115.513293, 20 fc rgb "#FFFF00"
set object 28 rect from 0.090412, 10 to 117.074628, 20 fc rgb "#FF00FF"
set object 29 rect from 0.091629, 10 to 118.712752, 20 fc rgb "#808080"
set object 30 rect from 0.092912, 10 to 119.503657, 20 fc rgb "#800080"
set object 31 rect from 0.093779, 10 to 120.266408, 20 fc rgb "#008080"
set object 32 rect from 0.094095, 10 to 154.752745, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.119849, 20 to 157.863902, 30 fc rgb "#FF0000"
set object 34 rect from 0.073275, 20 to 96.028582, 30 fc rgb "#00FF00"
set object 35 rect from 0.075412, 20 to 96.856604, 30 fc rgb "#0000FF"
set object 36 rect from 0.075794, 20 to 97.852274, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076573, 20 to 99.285634, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077693, 20 to 100.763784, 30 fc rgb "#808080"
set object 39 rect from 0.078856, 20 to 101.433110, 30 fc rgb "#800080"
set object 40 rect from 0.079644, 20 to 102.199701, 30 fc rgb "#008080"
set object 41 rect from 0.079976, 20 to 152.650060, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.118239, 30 to 156.053006, 40 fc rgb "#FF0000"
set object 43 rect from 0.057691, 30 to 76.079311, 40 fc rgb "#00FF00"
set object 44 rect from 0.059885, 30 to 76.996917, 40 fc rgb "#0000FF"
set object 45 rect from 0.060279, 30 to 78.217831, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061232, 30 to 79.681903, 40 fc rgb "#FF00FF"
set object 47 rect from 0.062377, 30 to 81.122941, 40 fc rgb "#808080"
set object 48 rect from 0.063523, 30 to 81.839618, 40 fc rgb "#800080"
set object 49 rect from 0.064333, 30 to 82.619008, 40 fc rgb "#008080"
set object 50 rect from 0.064675, 30 to 150.470589, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.116526, 40 to 154.147408, 50 fc rgb "#FF0000"
set object 52 rect from 0.042030, 40 to 56.565168, 50 fc rgb "#00FF00"
set object 53 rect from 0.044570, 40 to 57.408546, 50 fc rgb "#0000FF"
set object 54 rect from 0.044970, 40 to 58.564190, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045879, 40 to 60.165201, 50 fc rgb "#FF00FF"
set object 56 rect from 0.047139, 40 to 61.711179, 50 fc rgb "#808080"
set object 57 rect from 0.048345, 40 to 62.530240, 50 fc rgb "#800080"
set object 58 rect from 0.049260, 40 to 63.412011, 50 fc rgb "#008080"
set object 59 rect from 0.049697, 40 to 148.177215, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.114550, 50 to 154.544143, 60 fc rgb "#FF0000"
set object 61 rect from 0.021472, 50 to 30.993806, 60 fc rgb "#00FF00"
set object 62 rect from 0.024658, 50 to 32.324782, 60 fc rgb "#0000FF"
set object 63 rect from 0.025401, 50 to 35.013608, 60 fc rgb "#FFFF00"
set object 64 rect from 0.027515, 50 to 37.176441, 60 fc rgb "#FF00FF"
set object 65 rect from 0.029169, 50 to 39.046207, 60 fc rgb "#808080"
set object 66 rect from 0.030648, 50 to 40.031639, 60 fc rgb "#800080"
set object 67 rect from 0.031852, 50 to 41.473956, 60 fc rgb "#008080"
set object 68 rect from 0.032556, 50 to 145.300261, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
