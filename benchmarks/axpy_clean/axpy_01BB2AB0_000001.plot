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

set object 15 rect from 0.119954, 0 to 161.518153, 10 fc rgb "#FF0000"
set object 16 rect from 0.099121, 0 to 132.842041, 10 fc rgb "#00FF00"
set object 17 rect from 0.101372, 0 to 133.741623, 10 fc rgb "#0000FF"
set object 18 rect from 0.101807, 0 to 134.858207, 10 fc rgb "#FFFF00"
set object 19 rect from 0.102657, 0 to 136.210210, 10 fc rgb "#FF00FF"
set object 20 rect from 0.103682, 0 to 136.794147, 10 fc rgb "#808080"
set object 21 rect from 0.104139, 0 to 137.505659, 10 fc rgb "#800080"
set object 22 rect from 0.104948, 0 to 138.263200, 10 fc rgb "#008080"
set object 23 rect from 0.105252, 0 to 157.021564, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.118364, 10 to 159.705840, 20 fc rgb "#FF0000"
set object 25 rect from 0.084702, 10 to 113.903499, 20 fc rgb "#00FF00"
set object 26 rect from 0.086949, 10 to 114.742582, 20 fc rgb "#0000FF"
set object 27 rect from 0.087359, 10 to 115.902568, 20 fc rgb "#FFFF00"
set object 28 rect from 0.088273, 10 to 117.489985, 20 fc rgb "#FF00FF"
set object 29 rect from 0.089478, 10 to 118.188346, 20 fc rgb "#808080"
set object 30 rect from 0.090001, 10 to 118.951148, 20 fc rgb "#800080"
set object 31 rect from 0.090813, 10 to 119.711319, 20 fc rgb "#008080"
set object 32 rect from 0.091142, 10 to 154.868619, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.116593, 20 to 156.957124, 30 fc rgb "#FF0000"
set object 34 rect from 0.071522, 20 to 96.576048, 30 fc rgb "#00FF00"
set object 35 rect from 0.073796, 20 to 97.361208, 30 fc rgb "#0000FF"
set object 36 rect from 0.074149, 20 to 98.514619, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075042, 20 to 99.949476, 30 fc rgb "#FF00FF"
set object 38 rect from 0.076111, 20 to 100.480808, 30 fc rgb "#808080"
set object 39 rect from 0.076528, 20 to 101.171275, 30 fc rgb "#800080"
set object 40 rect from 0.077385, 20 to 102.086637, 30 fc rgb "#008080"
set object 41 rect from 0.077739, 20 to 152.723567, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.115043, 30 to 155.088255, 40 fc rgb "#FF0000"
set object 43 rect from 0.053179, 30 to 72.438612, 40 fc rgb "#00FF00"
set object 44 rect from 0.055460, 30 to 73.251390, 40 fc rgb "#0000FF"
set object 45 rect from 0.055815, 30 to 74.487657, 40 fc rgb "#FFFF00"
set object 46 rect from 0.056749, 30 to 75.888321, 40 fc rgb "#FF00FF"
set object 47 rect from 0.057821, 30 to 76.463052, 40 fc rgb "#808080"
set object 48 rect from 0.058256, 30 to 77.186399, 40 fc rgb "#800080"
set object 49 rect from 0.059068, 30 to 78.103077, 40 fc rgb "#008080"
set object 50 rect from 0.059506, 30 to 150.489081, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.113360, 40 to 153.076033, 50 fc rgb "#FF0000"
set object 52 rect from 0.039239, 40 to 54.344412, 50 fc rgb "#00FF00"
set object 53 rect from 0.041678, 40 to 55.159822, 50 fc rgb "#0000FF"
set object 54 rect from 0.042056, 40 to 56.425022, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043029, 40 to 57.974300, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044217, 40 to 58.621366, 50 fc rgb "#808080"
set object 57 rect from 0.044702, 40 to 59.317095, 50 fc rgb "#800080"
set object 58 rect from 0.045513, 40 to 60.271912, 50 fc rgb "#008080"
set object 59 rect from 0.045956, 40 to 148.216456, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.111500, 50 to 154.217607, 60 fc rgb "#FF0000"
set object 61 rect from 0.019287, 50 to 29.311341, 60 fc rgb "#00FF00"
set object 62 rect from 0.022873, 50 to 31.031591, 60 fc rgb "#0000FF"
set object 63 rect from 0.023727, 50 to 34.169603, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026148, 50 to 36.275202, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027715, 50 to 37.333919, 60 fc rgb "#808080"
set object 66 rect from 0.028540, 50 to 38.253228, 60 fc rgb "#800080"
set object 67 rect from 0.029638, 50 to 39.986629, 60 fc rgb "#008080"
set object 68 rect from 0.030539, 50 to 145.488780, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
