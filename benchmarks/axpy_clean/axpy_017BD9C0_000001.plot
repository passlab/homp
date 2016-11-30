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

set object 15 rect from 0.104139, 0 to 162.053030, 10 fc rgb "#FF0000"
set object 16 rect from 0.086474, 0 to 133.052499, 10 fc rgb "#00FF00"
set object 17 rect from 0.088182, 0 to 133.908116, 10 fc rgb "#0000FF"
set object 18 rect from 0.088533, 0 to 134.819787, 10 fc rgb "#FFFF00"
set object 19 rect from 0.089133, 0 to 136.179712, 10 fc rgb "#FF00FF"
set object 20 rect from 0.090033, 0 to 137.636535, 10 fc rgb "#808080"
set object 21 rect from 0.091017, 0 to 138.411907, 10 fc rgb "#800080"
set object 22 rect from 0.091758, 0 to 139.164532, 10 fc rgb "#008080"
set object 23 rect from 0.092005, 0 to 156.869232, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.102744, 10 to 160.399250, 20 fc rgb "#FF0000"
set object 25 rect from 0.073564, 10 to 113.588093, 20 fc rgb "#00FF00"
set object 26 rect from 0.075337, 10 to 114.540609, 20 fc rgb "#0000FF"
set object 27 rect from 0.075743, 10 to 115.503776, 20 fc rgb "#FFFF00"
set object 28 rect from 0.076411, 10 to 117.162025, 20 fc rgb "#FF00FF"
set object 29 rect from 0.077482, 10 to 118.649132, 20 fc rgb "#808080"
set object 30 rect from 0.078467, 10 to 119.465394, 20 fc rgb "#800080"
set object 31 rect from 0.079243, 10 to 120.307426, 20 fc rgb "#008080"
set object 32 rect from 0.079555, 10 to 154.661273, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.101200, 20 to 157.591573, 30 fc rgb "#FF0000"
set object 34 rect from 0.061625, 20 to 95.189801, 30 fc rgb "#00FF00"
set object 35 rect from 0.063180, 20 to 96.013600, 30 fc rgb "#0000FF"
set object 36 rect from 0.063508, 20 to 97.031287, 30 fc rgb "#FFFF00"
set object 37 rect from 0.064181, 20 to 98.497227, 30 fc rgb "#FF00FF"
set object 38 rect from 0.065150, 20 to 99.745043, 30 fc rgb "#808080"
set object 39 rect from 0.065973, 20 to 100.508319, 30 fc rgb "#800080"
set object 40 rect from 0.066727, 20 to 101.374542, 30 fc rgb "#008080"
set object 41 rect from 0.067050, 20 to 152.681953, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.099892, 30 to 155.771345, 40 fc rgb "#FF0000"
set object 43 rect from 0.048700, 30 to 75.870766, 40 fc rgb "#00FF00"
set object 44 rect from 0.050431, 30 to 76.668840, 40 fc rgb "#0000FF"
set object 45 rect from 0.050736, 30 to 77.837945, 40 fc rgb "#FFFF00"
set object 46 rect from 0.051523, 30 to 79.387154, 40 fc rgb "#FF00FF"
set object 47 rect from 0.052531, 30 to 80.592636, 40 fc rgb "#808080"
set object 48 rect from 0.053326, 30 to 81.392199, 40 fc rgb "#800080"
set object 49 rect from 0.054105, 30 to 82.294799, 40 fc rgb "#008080"
set object 50 rect from 0.054463, 30 to 150.626947, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.098607, 40 to 154.020620, 50 fc rgb "#FF0000"
set object 52 rect from 0.035400, 40 to 56.168604, 50 fc rgb "#00FF00"
set object 53 rect from 0.037418, 40 to 56.975749, 50 fc rgb "#0000FF"
set object 54 rect from 0.037732, 40 to 58.084287, 50 fc rgb "#FFFF00"
set object 55 rect from 0.038496, 40 to 59.762213, 50 fc rgb "#FF00FF"
set object 56 rect from 0.039598, 40 to 61.140326, 50 fc rgb "#808080"
set object 57 rect from 0.040498, 40 to 61.968683, 50 fc rgb "#800080"
set object 58 rect from 0.041289, 40 to 62.987859, 50 fc rgb "#008080"
set object 59 rect from 0.041723, 40 to 148.553752, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.097070, 50 to 154.673368, 60 fc rgb "#FF0000"
set object 61 rect from 0.017166, 50 to 30.172616, 60 fc rgb "#00FF00"
set object 62 rect from 0.020354, 50 to 31.723359, 60 fc rgb "#0000FF"
set object 63 rect from 0.021070, 50 to 34.462886, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022902, 50 to 36.652702, 60 fc rgb "#FF00FF"
set object 65 rect from 0.024323, 50 to 38.127714, 60 fc rgb "#808080"
set object 66 rect from 0.025323, 50 to 39.122654, 60 fc rgb "#800080"
set object 67 rect from 0.026395, 50 to 40.793044, 60 fc rgb "#008080"
set object 68 rect from 0.027061, 50 to 145.429564, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
