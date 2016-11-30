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

set object 15 rect from 0.095859, 0 to 162.632516, 10 fc rgb "#FF0000"
set object 16 rect from 0.078699, 0 to 131.852029, 10 fc rgb "#00FF00"
set object 17 rect from 0.080292, 0 to 132.824785, 10 fc rgb "#0000FF"
set object 18 rect from 0.080662, 0 to 133.850313, 10 fc rgb "#FFFF00"
set object 19 rect from 0.081285, 0 to 135.269867, 10 fc rgb "#FF00FF"
set object 20 rect from 0.082143, 0 to 136.733938, 10 fc rgb "#808080"
set object 21 rect from 0.083033, 0 to 137.469315, 10 fc rgb "#800080"
set object 22 rect from 0.083727, 0 to 138.270585, 10 fc rgb "#008080"
set object 23 rect from 0.083965, 0 to 157.209761, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.094566, 10 to 160.980448, 20 fc rgb "#FF0000"
set object 25 rect from 0.066994, 10 to 112.662256, 20 fc rgb "#00FF00"
set object 26 rect from 0.068634, 10 to 113.613588, 20 fc rgb "#0000FF"
set object 27 rect from 0.069020, 10 to 114.759452, 20 fc rgb "#FFFF00"
set object 28 rect from 0.069731, 10 to 116.464243, 20 fc rgb "#FF00FF"
set object 29 rect from 0.070769, 10 to 117.997597, 20 fc rgb "#808080"
set object 30 rect from 0.071696, 10 to 118.879647, 20 fc rgb "#800080"
set object 31 rect from 0.072422, 10 to 119.697426, 20 fc rgb "#008080"
set object 32 rect from 0.072703, 10 to 155.018568, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.093092, 20 to 158.047391, 30 fc rgb "#FF0000"
set object 34 rect from 0.056330, 20 to 94.931592, 30 fc rgb "#00FF00"
set object 35 rect from 0.057897, 20 to 95.833493, 30 fc rgb "#0000FF"
set object 36 rect from 0.058225, 20 to 96.850766, 30 fc rgb "#FFFF00"
set object 37 rect from 0.058849, 20 to 98.471485, 30 fc rgb "#FF00FF"
set object 38 rect from 0.059823, 20 to 99.770654, 30 fc rgb "#808080"
set object 39 rect from 0.060646, 20 to 100.580227, 30 fc rgb "#800080"
set object 40 rect from 0.061323, 20 to 101.435890, 30 fc rgb "#008080"
set object 41 rect from 0.061644, 20 to 152.802659, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.091826, 30 to 156.019331, 40 fc rgb "#FF0000"
set object 43 rect from 0.043902, 30 to 74.417885, 40 fc rgb "#00FF00"
set object 44 rect from 0.045475, 30 to 75.281852, 40 fc rgb "#0000FF"
set object 45 rect from 0.045761, 30 to 76.482110, 40 fc rgb "#FFFF00"
set object 46 rect from 0.046489, 30 to 78.064895, 40 fc rgb "#FF00FF"
set object 47 rect from 0.047461, 30 to 79.299843, 40 fc rgb "#808080"
set object 48 rect from 0.048217, 30 to 80.155506, 40 fc rgb "#800080"
set object 49 rect from 0.048932, 30 to 81.006256, 40 fc rgb "#008080"
set object 50 rect from 0.049248, 30 to 150.609794, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.090529, 40 to 154.195826, 50 fc rgb "#FF0000"
set object 52 rect from 0.031235, 40 to 54.027805, 50 fc rgb "#00FF00"
set object 53 rect from 0.033108, 40 to 54.962676, 50 fc rgb "#0000FF"
set object 54 rect from 0.033435, 40 to 56.118416, 50 fc rgb "#FFFF00"
set object 55 rect from 0.034159, 40 to 57.986488, 50 fc rgb "#FF00FF"
set object 56 rect from 0.035282, 40 to 59.279073, 50 fc rgb "#808080"
set object 57 rect from 0.036079, 40 to 60.118276, 50 fc rgb "#800080"
set object 58 rect from 0.036814, 40 to 61.105920, 50 fc rgb "#008080"
set object 59 rect from 0.037194, 40 to 148.428526, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.089088, 50 to 155.201601, 60 fc rgb "#FF0000"
set object 61 rect from 0.015313, 50 to 28.772326, 60 fc rgb "#00FF00"
set object 62 rect from 0.017823, 50 to 30.325433, 60 fc rgb "#0000FF"
set object 63 rect from 0.018504, 50 to 32.915615, 60 fc rgb "#FFFF00"
set object 64 rect from 0.020117, 50 to 35.736591, 60 fc rgb "#FF00FF"
set object 65 rect from 0.021797, 50 to 37.388659, 60 fc rgb "#808080"
set object 66 rect from 0.022800, 50 to 38.359793, 60 fc rgb "#800080"
set object 67 rect from 0.023767, 50 to 40.028322, 60 fc rgb "#008080"
set object 68 rect from 0.024394, 50 to 145.365160, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
