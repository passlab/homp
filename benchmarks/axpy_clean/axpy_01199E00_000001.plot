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

set object 15 rect from 0.108380, 0 to 160.224658, 10 fc rgb "#FF0000"
set object 16 rect from 0.091063, 0 to 133.923769, 10 fc rgb "#00FF00"
set object 17 rect from 0.092636, 0 to 134.725865, 10 fc rgb "#0000FF"
set object 18 rect from 0.092986, 0 to 135.498951, 10 fc rgb "#FFFF00"
set object 19 rect from 0.093543, 0 to 136.762288, 10 fc rgb "#FF00FF"
set object 20 rect from 0.094393, 0 to 137.216275, 10 fc rgb "#808080"
set object 21 rect from 0.094712, 0 to 137.874778, 10 fc rgb "#800080"
set object 22 rect from 0.095381, 0 to 138.552136, 10 fc rgb "#008080"
set object 23 rect from 0.095627, 0 to 156.505721, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.107026, 10 to 158.806126, 20 fc rgb "#FF0000"
set object 25 rect from 0.078790, 10 to 116.145689, 20 fc rgb "#00FF00"
set object 26 rect from 0.080391, 10 to 117.021756, 20 fc rgb "#0000FF"
set object 27 rect from 0.080788, 10 to 117.929733, 20 fc rgb "#FFFF00"
set object 28 rect from 0.081442, 10 to 119.380177, 20 fc rgb "#FF00FF"
set object 29 rect from 0.082415, 10 to 119.948749, 20 fc rgb "#808080"
set object 30 rect from 0.082814, 10 to 120.708783, 20 fc rgb "#800080"
set object 31 rect from 0.083561, 10 to 121.464462, 20 fc rgb "#008080"
set object 32 rect from 0.083847, 10 to 154.534568, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.105685, 20 to 156.524579, 30 fc rgb "#FF0000"
set object 34 rect from 0.067284, 20 to 99.147943, 30 fc rgb "#00FF00"
set object 35 rect from 0.068682, 20 to 99.987752, 30 fc rgb "#0000FF"
set object 36 rect from 0.069036, 20 to 100.888477, 30 fc rgb "#FFFF00"
set object 37 rect from 0.069660, 20 to 102.166317, 30 fc rgb "#FF00FF"
set object 38 rect from 0.070539, 20 to 102.650765, 30 fc rgb "#808080"
set object 39 rect from 0.070895, 20 to 103.328120, 30 fc rgb "#800080"
set object 40 rect from 0.071572, 20 to 104.102657, 30 fc rgb "#008080"
set object 41 rect from 0.071875, 20 to 152.592425, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.104256, 30 to 154.428684, 40 fc rgb "#FF0000"
set object 43 rect from 0.055015, 30 to 81.762933, 40 fc rgb "#00FF00"
set object 44 rect from 0.056686, 30 to 82.467848, 40 fc rgb "#0000FF"
set object 45 rect from 0.056961, 30 to 83.406283, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057621, 30 to 84.749395, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058542, 30 to 85.248346, 40 fc rgb "#808080"
set object 48 rect from 0.058885, 30 to 85.927154, 40 fc rgb "#800080"
set object 49 rect from 0.059586, 30 to 86.677033, 40 fc rgb "#008080"
set object 50 rect from 0.059861, 30 to 150.487831, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.102790, 40 to 152.582272, 50 fc rgb "#FF0000"
set object 52 rect from 0.042749, 40 to 64.396775, 50 fc rgb "#00FF00"
set object 53 rect from 0.044704, 40 to 65.317806, 50 fc rgb "#0000FF"
set object 54 rect from 0.045136, 40 to 66.240288, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045791, 40 to 67.638517, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046735, 40 to 68.136017, 50 fc rgb "#808080"
set object 57 rect from 0.047095, 40 to 68.832232, 50 fc rgb "#800080"
set object 58 rect from 0.047808, 40 to 69.674940, 50 fc rgb "#008080"
set object 59 rect from 0.048158, 40 to 148.336825, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.101284, 50 to 153.593231, 60 fc rgb "#FF0000"
set object 61 rect from 0.025097, 50 to 39.760998, 60 fc rgb "#00FF00"
set object 62 rect from 0.027831, 50 to 41.250602, 60 fc rgb "#0000FF"
set object 63 rect from 0.028563, 50 to 43.722158, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030297, 50 to 45.929733, 60 fc rgb "#FF00FF"
set object 65 rect from 0.031771, 50 to 46.707171, 60 fc rgb "#808080"
set object 66 rect from 0.032335, 50 to 47.635454, 60 fc rgb "#800080"
set object 67 rect from 0.033319, 50 to 49.300563, 60 fc rgb "#008080"
set object 68 rect from 0.034098, 50 to 145.306847, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
