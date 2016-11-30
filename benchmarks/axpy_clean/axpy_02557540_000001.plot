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

set object 15 rect from 7.070301, 0 to 145.754506, 10 fc rgb "#FF0000"
set object 16 rect from 0.370240, 0 to 8.164718, 10 fc rgb "#00FF00"
set object 17 rect from 0.404333, 0 to 8.370222, 10 fc rgb "#0000FF"
set object 18 rect from 0.410907, 0 to 8.592331, 10 fc rgb "#FFFF00"
set object 19 rect from 0.421852, 0 to 8.843952, 10 fc rgb "#FF00FF"
set object 20 rect from 0.434077, 0 to 9.565443, 10 fc rgb "#808080"
set object 21 rect from 0.469511, 0 to 9.679326, 10 fc rgb "#800080"
set object 22 rect from 0.478293, 0 to 9.805749, 10 fc rgb "#008080"
set object 23 rect from 0.481211, 0 to 144.243309, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 7.087153, 10 to 145.967955, 20 fc rgb "#FF0000"
set object 25 rect from 1.423682, 10 to 29.561011, 20 fc rgb "#00FF00"
set object 26 rect from 1.451073, 10 to 29.710738, 20 fc rgb "#0000FF"
set object 27 rect from 1.455661, 10 to 29.887506, 20 fc rgb "#FFFF00"
set object 28 rect from 1.464499, 10 to 30.095502, 20 fc rgb "#FF00FF"
set object 29 rect from 1.474547, 10 to 30.823426, 20 fc rgb "#808080"
set object 30 rect from 1.510553, 10 to 30.929569, 20 fc rgb "#800080"
set object 31 rect from 1.518318, 10 to 31.044187, 20 fc rgb "#008080"
set object 32 rect from 1.521042, 10 to 144.635956, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 7.101433, 20 to 146.879449, 30 fc rgb "#FF0000"
set object 34 rect from 6.348746, 20 to 130.148767, 30 fc rgb "#00FF00"
set object 35 rect from 6.376275, 20 to 130.316058, 30 fc rgb "#0000FF"
set object 36 rect from 6.381691, 20 to 130.897095, 30 fc rgb "#FFFF00"
set object 37 rect from 6.410214, 20 to 131.241276, 30 fc rgb "#FF00FF"
set object 38 rect from 6.426880, 20 to 132.018933, 30 fc rgb "#808080"
set object 39 rect from 6.465072, 20 to 132.140189, 30 fc rgb "#800080"
set object 40 rect from 6.474290, 20 to 132.330355, 30 fc rgb "#008080"
set object 41 rect from 6.480387, 20 to 144.931917, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 7.184594, 30 to 148.272637, 40 fc rgb "#FF0000"
set object 43 rect from 6.568581, 30 to 134.490130, 40 fc rgb "#00FF00"
set object 44 rect from 6.587765, 30 to 134.588634, 40 fc rgb "#0000FF"
set object 45 rect from 6.590719, 30 to 135.044881, 40 fc rgb "#FFFF00"
set object 46 rect from 6.613043, 30 to 135.306326, 40 fc rgb "#FF00FF"
set object 47 rect from 6.626028, 30 to 136.047016, 40 fc rgb "#808080"
set object 48 rect from 6.662185, 30 to 136.140618, 40 fc rgb "#800080"
set object 49 rect from 6.669050, 30 to 136.247475, 40 fc rgb "#008080"
set object 50 rect from 6.672129, 30 to 145.208476, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 7.207376, 40 to 148.724514, 50 fc rgb "#FF0000"
set object 52 rect from 6.724744, 40 to 137.633720, 50 fc rgb "#00FF00"
set object 53 rect from 6.741938, 40 to 137.746643, 50 fc rgb "#0000FF"
set object 54 rect from 6.745333, 40 to 138.210957, 50 fc rgb "#FFFF00"
set object 55 rect from 6.768046, 40 to 138.449140, 50 fc rgb "#FF00FF"
set object 56 rect from 6.779749, 40 to 139.180026, 50 fc rgb "#808080"
set object 57 rect from 6.815548, 40 to 139.276508, 50 fc rgb "#800080"
set object 58 rect from 6.822257, 40 to 139.369272, 50 fc rgb "#008080"
set object 59 rect from 6.824896, 40 to 147.124534, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 7.220810, 50 to 148.936533, 60 fc rgb "#FF0000"
set object 61 rect from 6.905133, 50 to 141.309655, 60 fc rgb "#00FF00"
set object 62 rect from 6.921762, 50 to 141.405831, 60 fc rgb "#0000FF"
set object 63 rect from 6.924432, 50 to 141.844269, 60 fc rgb "#FFFF00"
set object 64 rect from 6.945902, 50 to 142.071259, 60 fc rgb "#FF00FF"
set object 65 rect from 6.957075, 50 to 142.790809, 60 fc rgb "#808080"
set object 66 rect from 6.992366, 50 to 142.884616, 60 fc rgb "#800080"
set object 67 rect from 6.998977, 50 to 142.979934, 60 fc rgb "#008080"
set object 68 rect from 7.001521, 50 to 147.395578, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
