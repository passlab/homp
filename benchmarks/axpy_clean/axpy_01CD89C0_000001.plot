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

set object 15 rect from 0.102293, 0 to 161.038721, 10 fc rgb "#FF0000"
set object 16 rect from 0.085253, 0 to 132.758501, 10 fc rgb "#00FF00"
set object 17 rect from 0.086845, 0 to 133.603742, 10 fc rgb "#0000FF"
set object 18 rect from 0.087193, 0 to 134.450537, 10 fc rgb "#FFFF00"
set object 19 rect from 0.087744, 0 to 135.757533, 10 fc rgb "#FF00FF"
set object 20 rect from 0.088597, 0 to 137.119802, 10 fc rgb "#808080"
set object 21 rect from 0.089487, 0 to 137.808594, 10 fc rgb "#800080"
set object 22 rect from 0.090151, 0 to 138.506576, 10 fc rgb "#008080"
set object 23 rect from 0.090391, 0 to 156.189605, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.100981, 10 to 159.633568, 20 fc rgb "#FF0000"
set object 25 rect from 0.073179, 10 to 114.374382, 20 fc rgb "#00FF00"
set object 26 rect from 0.074860, 10 to 115.297892, 20 fc rgb "#0000FF"
set object 27 rect from 0.075260, 10 to 116.302690, 20 fc rgb "#FFFF00"
set object 28 rect from 0.075949, 10 to 117.891996, 20 fc rgb "#FF00FF"
set object 29 rect from 0.076969, 10 to 119.376928, 20 fc rgb "#808080"
set object 30 rect from 0.077929, 10 to 120.124011, 20 fc rgb "#800080"
set object 31 rect from 0.078636, 10 to 120.909452, 20 fc rgb "#008080"
set object 32 rect from 0.078921, 10 to 154.034260, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.099479, 20 to 156.981218, 30 fc rgb "#FF0000"
set object 34 rect from 0.062138, 20 to 97.142364, 30 fc rgb "#00FF00"
set object 35 rect from 0.063635, 20 to 97.960036, 30 fc rgb "#0000FF"
set object 36 rect from 0.063957, 20 to 98.952582, 30 fc rgb "#FFFF00"
set object 37 rect from 0.064605, 20 to 100.643063, 30 fc rgb "#FF00FF"
set object 38 rect from 0.065705, 20 to 101.729193, 30 fc rgb "#808080"
set object 39 rect from 0.066417, 20 to 102.460915, 30 fc rgb "#800080"
set object 40 rect from 0.067109, 20 to 103.220297, 30 fc rgb "#008080"
set object 41 rect from 0.067387, 20 to 152.056851, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.098271, 30 to 155.255351, 40 fc rgb "#FF0000"
set object 43 rect from 0.049861, 30 to 78.425415, 40 fc rgb "#00FF00"
set object 44 rect from 0.051436, 30 to 79.276782, 40 fc rgb "#0000FF"
set object 45 rect from 0.051779, 30 to 80.445754, 40 fc rgb "#FFFF00"
set object 46 rect from 0.052564, 30 to 82.013527, 40 fc rgb "#FF00FF"
set object 47 rect from 0.053560, 30 to 83.145696, 40 fc rgb "#808080"
set object 48 rect from 0.054299, 30 to 83.906586, 40 fc rgb "#800080"
set object 49 rect from 0.055023, 30 to 84.708897, 40 fc rgb "#008080"
set object 50 rect from 0.055336, 30 to 150.048810, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.097023, 40 to 154.083362, 50 fc rgb "#FF0000"
set object 52 rect from 0.037033, 40 to 59.168441, 50 fc rgb "#00FF00"
set object 53 rect from 0.038883, 40 to 60.343539, 50 fc rgb "#0000FF"
set object 54 rect from 0.039446, 40 to 61.372887, 50 fc rgb "#FFFF00"
set object 55 rect from 0.040107, 40 to 63.449963, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041486, 40 to 64.623507, 50 fc rgb "#808080"
set object 57 rect from 0.042234, 40 to 65.421246, 50 fc rgb "#800080"
set object 58 rect from 0.043004, 40 to 66.315588, 50 fc rgb "#008080"
set object 59 rect from 0.043376, 40 to 148.099014, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.095596, 50 to 154.344139, 60 fc rgb "#FF0000"
set object 61 rect from 0.018695, 50 to 32.402276, 60 fc rgb "#00FF00"
set object 62 rect from 0.021523, 50 to 33.914867, 60 fc rgb "#0000FF"
set object 63 rect from 0.022221, 50 to 36.538093, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023965, 50 to 38.891308, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025470, 50 to 40.299569, 60 fc rgb "#808080"
set object 66 rect from 0.026394, 50 to 41.312048, 60 fc rgb "#800080"
set object 67 rect from 0.027478, 50 to 43.048613, 60 fc rgb "#008080"
set object 68 rect from 0.028179, 50 to 145.403690, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
