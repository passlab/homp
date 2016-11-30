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

set object 15 rect from 0.124340, 0 to 160.825310, 10 fc rgb "#FF0000"
set object 16 rect from 0.103418, 0 to 133.234793, 10 fc rgb "#00FF00"
set object 17 rect from 0.105454, 0 to 133.983909, 10 fc rgb "#0000FF"
set object 18 rect from 0.105819, 0 to 134.897805, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106539, 0 to 136.189426, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107566, 0 to 136.710384, 10 fc rgb "#808080"
set object 21 rect from 0.107973, 0 to 137.337818, 10 fc rgb "#800080"
set object 22 rect from 0.108713, 0 to 137.974122, 10 fc rgb "#008080"
set object 23 rect from 0.108972, 0 to 156.879460, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.122771, 10 to 159.548898, 20 fc rgb "#FF0000"
set object 25 rect from 0.088588, 10 to 114.775628, 20 fc rgb "#00FF00"
set object 26 rect from 0.090902, 10 to 115.727549, 20 fc rgb "#0000FF"
set object 27 rect from 0.091429, 10 to 116.740313, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092254, 10 to 118.300653, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093461, 10 to 118.944564, 20 fc rgb "#808080"
set object 30 rect from 0.093970, 10 to 119.701284, 20 fc rgb "#800080"
set object 31 rect from 0.094815, 10 to 120.447865, 20 fc rgb "#008080"
set object 32 rect from 0.095144, 10 to 154.812105, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121072, 20 to 157.106347, 30 fc rgb "#FF0000"
set object 34 rect from 0.074989, 20 to 97.277256, 30 fc rgb "#00FF00"
set object 35 rect from 0.077101, 20 to 98.115100, 30 fc rgb "#0000FF"
set object 36 rect from 0.077540, 20 to 99.230533, 30 fc rgb "#FFFF00"
set object 37 rect from 0.078399, 20 to 100.628629, 30 fc rgb "#FF00FF"
set object 38 rect from 0.079501, 20 to 101.149587, 30 fc rgb "#808080"
set object 39 rect from 0.079916, 20 to 101.794764, 30 fc rgb "#800080"
set object 40 rect from 0.080692, 20 to 102.533740, 30 fc rgb "#008080"
set object 41 rect from 0.081010, 20 to 152.733343, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119454, 30 to 155.173351, 40 fc rgb "#FF0000"
set object 43 rect from 0.060534, 30 to 79.118498, 40 fc rgb "#00FF00"
set object 44 rect from 0.062782, 30 to 80.000705, 40 fc rgb "#0000FF"
set object 45 rect from 0.063243, 30 to 81.135151, 40 fc rgb "#FFFF00"
set object 46 rect from 0.064147, 30 to 82.625776, 40 fc rgb "#FF00FF"
set object 47 rect from 0.065298, 30 to 83.130257, 40 fc rgb "#808080"
set object 48 rect from 0.065710, 30 to 83.835008, 40 fc rgb "#800080"
set object 49 rect from 0.066549, 30 to 84.618349, 40 fc rgb "#008080"
set object 50 rect from 0.066872, 30 to 150.704011, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.117740, 40 to 153.152895, 50 fc rgb "#FF0000"
set object 52 rect from 0.045237, 40 to 59.859515, 50 fc rgb "#00FF00"
set object 53 rect from 0.047607, 40 to 60.715104, 50 fc rgb "#0000FF"
set object 54 rect from 0.048030, 40 to 61.878703, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048944, 40 to 63.413693, 50 fc rgb "#FF00FF"
set object 56 rect from 0.050159, 40 to 64.048729, 50 fc rgb "#808080"
set object 57 rect from 0.050685, 40 to 64.794042, 50 fc rgb "#800080"
set object 58 rect from 0.051531, 40 to 65.616676, 50 fc rgb "#008080"
set object 59 rect from 0.051921, 40 to 148.177809, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.115718, 50 to 155.028852, 60 fc rgb "#FF0000"
set object 61 rect from 0.022749, 50 to 32.445188, 60 fc rgb "#00FF00"
set object 62 rect from 0.026051, 50 to 33.867366, 60 fc rgb "#0000FF"
set object 63 rect from 0.026858, 50 to 38.047708, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030180, 50 to 40.137879, 60 fc rgb "#FF00FF"
set object 65 rect from 0.031810, 50 to 41.015017, 60 fc rgb "#808080"
set object 66 rect from 0.032502, 50 to 41.944124, 60 fc rgb "#800080"
set object 67 rect from 0.033694, 50 to 43.231941, 60 fc rgb "#008080"
set object 68 rect from 0.034260, 50 to 145.433586, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
