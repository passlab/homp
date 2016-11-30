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

set object 15 rect from 0.137983, 0 to 152.790719, 10 fc rgb "#FF0000"
set object 16 rect from 0.055744, 0 to 61.868940, 10 fc rgb "#00FF00"
set object 17 rect from 0.057985, 0 to 62.616449, 10 fc rgb "#0000FF"
set object 18 rect from 0.058415, 0 to 63.653932, 10 fc rgb "#FFFF00"
set object 19 rect from 0.059383, 0 to 64.749413, 10 fc rgb "#FF00FF"
set object 20 rect from 0.060422, 0 to 66.708391, 10 fc rgb "#808080"
set object 21 rect from 0.062245, 0 to 67.284059, 10 fc rgb "#800080"
set object 22 rect from 0.063061, 0 to 67.887645, 10 fc rgb "#008080"
set object 23 rect from 0.063336, 0 to 147.548499, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.136258, 10 to 153.785224, 20 fc rgb "#FF0000"
set object 25 rect from 0.034213, 10 to 40.132216, 20 fc rgb "#00FF00"
set object 26 rect from 0.037827, 10 to 41.510165, 20 fc rgb "#0000FF"
set object 27 rect from 0.038775, 10 to 43.267231, 20 fc rgb "#FFFF00"
set object 28 rect from 0.040451, 10 to 44.828825, 20 fc rgb "#FF00FF"
set object 29 rect from 0.041862, 10 to 47.833888, 20 fc rgb "#808080"
set object 30 rect from 0.044685, 10 to 48.535210, 20 fc rgb "#800080"
set object 31 rect from 0.045737, 10 to 49.380446, 20 fc rgb "#008080"
set object 32 rect from 0.046107, 10 to 145.045005, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.139880, 20 to 156.621654, 30 fc rgb "#FF0000"
set object 34 rect from 0.070840, 20 to 77.859754, 30 fc rgb "#00FF00"
set object 35 rect from 0.072854, 20 to 78.796285, 30 fc rgb "#0000FF"
set object 36 rect from 0.073479, 20 to 81.045237, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075604, 20 to 82.603614, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077037, 20 to 84.314500, 30 fc rgb "#808080"
set object 39 rect from 0.078641, 20 to 85.071675, 30 fc rgb "#800080"
set object 40 rect from 0.079585, 20 to 86.186489, 30 fc rgb "#008080"
set object 41 rect from 0.080376, 20 to 149.296978, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.141413, 30 to 157.428234, 40 fc rgb "#FF0000"
set object 43 rect from 0.089307, 30 to 97.472100, 40 fc rgb "#00FF00"
set object 44 rect from 0.091100, 30 to 98.247528, 40 fc rgb "#0000FF"
set object 45 rect from 0.091588, 30 to 100.302089, 40 fc rgb "#FFFF00"
set object 46 rect from 0.093501, 30 to 101.554376, 40 fc rgb "#FF00FF"
set object 47 rect from 0.094668, 30 to 103.042944, 40 fc rgb "#808080"
set object 48 rect from 0.096070, 30 to 103.764671, 40 fc rgb "#800080"
set object 49 rect from 0.096987, 30 to 104.511108, 40 fc rgb "#008080"
set object 50 rect from 0.097425, 30 to 151.272064, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.142870, 40 to 158.968366, 50 fc rgb "#FF0000"
set object 52 rect from 0.105192, 40 to 114.360771, 50 fc rgb "#00FF00"
set object 53 rect from 0.106829, 40 to 115.072840, 50 fc rgb "#0000FF"
set object 54 rect from 0.107260, 40 to 117.159625, 50 fc rgb "#FFFF00"
set object 55 rect from 0.109197, 40 to 118.499981, 50 fc rgb "#FF00FF"
set object 56 rect from 0.110462, 40 to 119.997135, 50 fc rgb "#808080"
set object 57 rect from 0.111852, 40 to 120.653350, 50 fc rgb "#800080"
set object 58 rect from 0.112706, 40 to 121.333195, 50 fc rgb "#008080"
set object 59 rect from 0.113091, 40 to 152.953951, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.144300, 50 to 160.194879, 60 fc rgb "#FF0000"
set object 61 rect from 0.122132, 50 to 132.521070, 60 fc rgb "#00FF00"
set object 62 rect from 0.123726, 50 to 133.085999, 60 fc rgb "#0000FF"
set object 63 rect from 0.124026, 50 to 135.140560, 60 fc rgb "#FFFF00"
set object 64 rect from 0.125941, 50 to 136.334849, 60 fc rgb "#FF00FF"
set object 65 rect from 0.127053, 50 to 137.812679, 60 fc rgb "#808080"
set object 66 rect from 0.128435, 50 to 138.466741, 60 fc rgb "#800080"
set object 67 rect from 0.129303, 50 to 139.105776, 60 fc rgb "#008080"
set object 68 rect from 0.129636, 50 to 154.535958, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
