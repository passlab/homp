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

set object 15 rect from 0.113778, 0 to 162.096925, 10 fc rgb "#FF0000"
set object 16 rect from 0.094057, 0 to 132.592856, 10 fc rgb "#00FF00"
set object 17 rect from 0.095963, 0 to 133.401511, 10 fc rgb "#0000FF"
set object 18 rect from 0.096297, 0 to 134.294773, 10 fc rgb "#FFFF00"
set object 19 rect from 0.096957, 0 to 135.749790, 10 fc rgb "#FF00FF"
set object 20 rect from 0.097991, 0 to 137.182620, 10 fc rgb "#808080"
set object 21 rect from 0.099044, 0 to 137.944119, 10 fc rgb "#800080"
set object 22 rect from 0.099857, 0 to 138.744444, 10 fc rgb "#008080"
set object 23 rect from 0.100167, 0 to 157.030012, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.112213, 10 to 160.598906, 20 fc rgb "#FF0000"
set object 25 rect from 0.079592, 10 to 112.645603, 20 fc rgb "#00FF00"
set object 26 rect from 0.081580, 10 to 113.615154, 20 fc rgb "#0000FF"
set object 27 rect from 0.082030, 10 to 114.588869, 20 fc rgb "#FFFF00"
set object 28 rect from 0.082783, 10 to 116.289400, 20 fc rgb "#FF00FF"
set object 29 rect from 0.083988, 10 to 117.906709, 20 fc rgb "#808080"
set object 30 rect from 0.085164, 10 to 118.772223, 20 fc rgb "#800080"
set object 31 rect from 0.086029, 10 to 119.659946, 20 fc rgb "#008080"
set object 32 rect from 0.086394, 10 to 154.773275, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.110530, 20 to 157.852536, 30 fc rgb "#FF0000"
set object 34 rect from 0.066316, 20 to 93.910634, 30 fc rgb "#00FF00"
set object 35 rect from 0.068079, 20 to 94.821930, 30 fc rgb "#0000FF"
set object 36 rect from 0.068482, 20 to 95.863603, 30 fc rgb "#FFFF00"
set object 37 rect from 0.069244, 20 to 97.396305, 30 fc rgb "#FF00FF"
set object 38 rect from 0.070355, 20 to 98.761166, 30 fc rgb "#808080"
set object 39 rect from 0.071345, 20 to 99.544852, 30 fc rgb "#800080"
set object 40 rect from 0.072162, 20 to 100.449213, 30 fc rgb "#008080"
set object 41 rect from 0.072543, 20 to 152.641366, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.109067, 30 to 155.993867, 40 fc rgb "#FF0000"
set object 43 rect from 0.052197, 30 to 74.523752, 40 fc rgb "#00FF00"
set object 44 rect from 0.054105, 30 to 75.405915, 40 fc rgb "#0000FF"
set object 45 rect from 0.054486, 30 to 76.611265, 40 fc rgb "#FFFF00"
set object 46 rect from 0.055371, 30 to 78.193902, 40 fc rgb "#FF00FF"
set object 47 rect from 0.056493, 30 to 79.519926, 40 fc rgb "#808080"
set object 48 rect from 0.057448, 30 to 80.284194, 40 fc rgb "#800080"
set object 49 rect from 0.058300, 30 to 81.189940, 40 fc rgb "#008080"
set object 50 rect from 0.058657, 30 to 150.556623, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.107563, 40 to 154.045063, 50 fc rgb "#FF0000"
set object 52 rect from 0.038497, 40 to 55.903908, 50 fc rgb "#00FF00"
set object 53 rect from 0.040667, 40 to 56.751399, 50 fc rgb "#0000FF"
set object 54 rect from 0.041035, 40 to 57.983101, 50 fc rgb "#FFFF00"
set object 55 rect from 0.041963, 40 to 59.701666, 50 fc rgb "#FF00FF"
set object 56 rect from 0.043192, 40 to 61.017987, 50 fc rgb "#808080"
set object 57 rect from 0.044139, 40 to 61.897380, 50 fc rgb "#800080"
set object 58 rect from 0.045031, 40 to 62.792027, 50 fc rgb "#008080"
set object 59 rect from 0.045409, 40 to 148.172266, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.105727, 50 to 154.516662, 60 fc rgb "#FF0000"
set object 61 rect from 0.019740, 50 to 30.992373, 60 fc rgb "#00FF00"
set object 62 rect from 0.022981, 50 to 32.801095, 60 fc rgb "#0000FF"
set object 63 rect from 0.023782, 50 to 35.539146, 60 fc rgb "#FFFF00"
set object 64 rect from 0.025796, 50 to 37.790345, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027380, 50 to 39.396555, 60 fc rgb "#808080"
set object 66 rect from 0.028555, 50 to 40.385523, 60 fc rgb "#800080"
set object 67 rect from 0.029687, 50 to 42.115177, 60 fc rgb "#008080"
set object 68 rect from 0.030503, 50 to 145.345453, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
