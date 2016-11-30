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

set object 15 rect from 0.141078, 0 to 157.897379, 10 fc rgb "#FF0000"
set object 16 rect from 0.103432, 0 to 114.800346, 10 fc rgb "#00FF00"
set object 17 rect from 0.105690, 0 to 115.663138, 10 fc rgb "#0000FF"
set object 18 rect from 0.106153, 0 to 116.491025, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106934, 0 to 117.665774, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108004, 0 to 119.076127, 10 fc rgb "#808080"
set object 21 rect from 0.109310, 0 to 119.739309, 10 fc rgb "#800080"
set object 22 rect from 0.110177, 0 to 120.350134, 10 fc rgb "#008080"
set object 23 rect from 0.110455, 0 to 153.185293, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.137748, 10 to 154.614188, 20 fc rgb "#FF0000"
set object 25 rect from 0.071232, 10 to 79.849112, 20 fc rgb "#00FF00"
set object 26 rect from 0.073672, 10 to 80.796982, 20 fc rgb "#0000FF"
set object 27 rect from 0.074189, 10 to 81.729582, 20 fc rgb "#FFFF00"
set object 28 rect from 0.075086, 10 to 83.072309, 20 fc rgb "#FF00FF"
set object 29 rect from 0.076311, 10 to 84.554652, 20 fc rgb "#808080"
set object 30 rect from 0.077666, 10 to 85.217835, 20 fc rgb "#800080"
set object 31 rect from 0.078543, 10 to 85.888652, 20 fc rgb "#008080"
set object 32 rect from 0.078879, 10 to 149.503978, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.135942, 20 to 152.459937, 30 fc rgb "#FF0000"
set object 34 rect from 0.055795, 20 to 63.137572, 30 fc rgb "#00FF00"
set object 35 rect from 0.058250, 20 to 63.854202, 30 fc rgb "#0000FF"
set object 36 rect from 0.058652, 20 to 64.726810, 30 fc rgb "#FFFF00"
set object 37 rect from 0.059460, 20 to 66.160070, 30 fc rgb "#FF00FF"
set object 38 rect from 0.060785, 20 to 67.477708, 30 fc rgb "#808080"
set object 39 rect from 0.062002, 20 to 68.138708, 30 fc rgb "#800080"
set object 40 rect from 0.062905, 20 to 68.898968, 30 fc rgb "#008080"
set object 41 rect from 0.063300, 20 to 147.632233, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.134145, 30 to 153.859383, 40 fc rgb "#FF0000"
set object 43 rect from 0.033523, 30 to 39.985093, 40 fc rgb "#00FF00"
set object 44 rect from 0.037102, 30 to 41.320184, 40 fc rgb "#0000FF"
set object 45 rect from 0.038007, 30 to 43.794029, 40 fc rgb "#FFFF00"
set object 46 rect from 0.040320, 30 to 46.030087, 40 fc rgb "#FF00FF"
set object 47 rect from 0.042336, 30 to 47.643321, 40 fc rgb "#808080"
set object 48 rect from 0.043920, 30 to 48.595555, 40 fc rgb "#800080"
set object 49 rect from 0.045196, 30 to 49.996092, 40 fc rgb "#008080"
set object 50 rect from 0.045977, 30 to 145.199836, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.139401, 40 to 156.309230, 50 fc rgb "#FF0000"
set object 52 rect from 0.087508, 40 to 97.519429, 50 fc rgb "#00FF00"
set object 53 rect from 0.089807, 40 to 98.317866, 50 fc rgb "#0000FF"
set object 54 rect from 0.090251, 40 to 99.309367, 50 fc rgb "#FFFF00"
set object 55 rect from 0.091175, 40 to 100.603009, 50 fc rgb "#FF00FF"
set object 56 rect from 0.092362, 40 to 101.914103, 50 fc rgb "#808080"
set object 57 rect from 0.093562, 40 to 102.634005, 50 fc rgb "#800080"
set object 58 rect from 0.094493, 40 to 103.346271, 50 fc rgb "#008080"
set object 59 rect from 0.094865, 40 to 151.394265, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.142629, 50 to 159.667681, 60 fc rgb "#FF0000"
set object 61 rect from 0.119349, 50 to 132.193610, 60 fc rgb "#00FF00"
set object 62 rect from 0.121548, 50 to 132.858974, 60 fc rgb "#0000FF"
set object 63 rect from 0.121915, 50 to 133.792665, 60 fc rgb "#FFFF00"
set object 64 rect from 0.122777, 50 to 135.059038, 60 fc rgb "#FF00FF"
set object 65 rect from 0.123951, 50 to 136.393038, 60 fc rgb "#808080"
set object 66 rect from 0.125192, 50 to 137.100941, 60 fc rgb "#800080"
set object 67 rect from 0.126093, 50 to 137.800119, 60 fc rgb "#008080"
set object 68 rect from 0.126455, 50 to 155.004682, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
