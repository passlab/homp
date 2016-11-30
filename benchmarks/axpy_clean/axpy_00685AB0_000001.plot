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

set object 15 rect from 0.788710, 0 to 153.489496, 10 fc rgb "#FF0000"
set object 16 rect from 0.727535, 0 to 135.592989, 10 fc rgb "#00FF00"
set object 17 rect from 0.729861, 0 to 135.730344, 10 fc rgb "#0000FF"
set object 18 rect from 0.730383, 0 to 135.869555, 10 fc rgb "#FFFF00"
set object 19 rect from 0.731158, 0 to 136.072521, 10 fc rgb "#FF00FF"
set object 20 rect from 0.732225, 0 to 142.517365, 10 fc rgb "#808080"
set object 21 rect from 0.766936, 0 to 142.630741, 10 fc rgb "#800080"
set object 22 rect from 0.767761, 0 to 142.738356, 10 fc rgb "#008080"
set object 23 rect from 0.768081, 0 to 146.474787, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.780601, 10 to 204.635429, 20 fc rgb "#FF0000"
set object 25 rect from 0.135339, 10 to 25.655212, 20 fc rgb "#00FF00"
set object 26 rect from 0.138394, 10 to 25.789039, 20 fc rgb "#0000FF"
set object 27 rect from 0.138858, 10 to 25.960405, 20 fc rgb "#FFFF00"
set object 28 rect from 0.139830, 10 to 26.208344, 20 fc rgb "#FF00FF"
set object 29 rect from 0.141117, 10 to 85.180472, 20 fc rgb "#808080"
set object 30 rect from 0.458667, 10 to 85.387526, 20 fc rgb "#800080"
set object 31 rect from 0.459933, 10 to 85.531385, 20 fc rgb "#008080"
set object 32 rect from 0.460377, 10 to 144.896611, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.783012, 20 to 158.012491, 30 fc rgb "#FF0000"
set object 34 rect from 0.477205, 20 to 89.177294, 30 fc rgb "#00FF00"
set object 35 rect from 0.480149, 20 to 89.356841, 30 fc rgb "#0000FF"
set object 36 rect from 0.480879, 20 to 90.053648, 30 fc rgb "#FFFF00"
set object 37 rect from 0.484660, 20 to 95.026071, 30 fc rgb "#FF00FF"
set object 38 rect from 0.511386, 20 to 101.427049, 30 fc rgb "#808080"
set object 39 rect from 0.545864, 20 to 101.796181, 30 fc rgb "#800080"
set object 40 rect from 0.548101, 20 to 101.983527, 30 fc rgb "#008080"
set object 41 rect from 0.548844, 20 to 145.337293, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.785176, 30 to 157.841490, 40 fc rgb "#FF0000"
set object 43 rect from 0.566194, 30 to 105.597104, 40 fc rgb "#00FF00"
set object 44 rect from 0.568484, 30 to 105.765307, 40 fc rgb "#0000FF"
set object 45 rect from 0.569166, 30 to 106.337211, 40 fc rgb "#FFFF00"
set object 46 rect from 0.572248, 30 to 111.011514, 40 fc rgb "#FF00FF"
set object 47 rect from 0.597414, 30 to 117.302091, 40 fc rgb "#808080"
set object 48 rect from 0.631329, 30 to 117.657646, 40 fc rgb "#800080"
set object 49 rect from 0.633442, 30 to 117.809869, 40 fc rgb "#008080"
set object 50 rect from 0.633965, 30 to 145.796380, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.777965, 40 to 160.711803, 50 fc rgb "#FF0000"
set object 52 rect from 0.026150, 40 to 5.464225, 50 fc rgb "#00FF00"
set object 53 rect from 0.029792, 40 to 5.657337, 50 fc rgb "#0000FF"
set object 54 rect from 0.030550, 40 to 6.277565, 50 fc rgb "#FFFF00"
set object 55 rect from 0.033918, 40 to 15.068016, 50 fc rgb "#FF00FF"
set object 56 rect from 0.081190, 40 to 21.353203, 50 fc rgb "#808080"
set object 57 rect from 0.115067, 40 to 21.733662, 50 fc rgb "#800080"
set object 58 rect from 0.117604, 40 to 21.986066, 50 fc rgb "#008080"
set object 59 rect from 0.118444, 40 to 144.371358, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.787050, 50 to 158.950369, 60 fc rgb "#FF0000"
set object 61 rect from 0.643599, 50 to 120.022768, 60 fc rgb "#00FF00"
set object 62 rect from 0.646150, 50 to 120.169047, 60 fc rgb "#0000FF"
set object 63 rect from 0.646667, 50 to 120.807117, 60 fc rgb "#FFFF00"
set object 64 rect from 0.650101, 50 to 126.227659, 60 fc rgb "#FF00FF"
set object 65 rect from 0.679256, 50 to 132.470472, 60 fc rgb "#808080"
set object 66 rect from 0.712899, 50 to 132.837365, 60 fc rgb "#800080"
set object 67 rect from 0.715145, 50 to 132.995537, 60 fc rgb "#008080"
set object 68 rect from 0.715676, 50 to 146.185392, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
