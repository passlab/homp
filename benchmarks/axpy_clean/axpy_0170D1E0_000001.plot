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

set object 15 rect from 0.179317, 0 to 158.671151, 10 fc rgb "#FF0000"
set object 16 rect from 0.103300, 0 to 88.076343, 10 fc rgb "#00FF00"
set object 17 rect from 0.105552, 0 to 88.787739, 10 fc rgb "#0000FF"
set object 18 rect from 0.106071, 0 to 89.418693, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106840, 0 to 90.298509, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107898, 0 to 96.726198, 10 fc rgb "#808080"
set object 21 rect from 0.115549, 0 to 97.156052, 10 fc rgb "#800080"
set object 22 rect from 0.116312, 0 to 97.575849, 10 fc rgb "#008080"
set object 23 rect from 0.116564, 0 to 149.743107, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.177606, 10 to 157.639671, 20 fc rgb "#FF0000"
set object 25 rect from 0.081217, 10 to 69.803810, 20 fc rgb "#00FF00"
set object 26 rect from 0.083672, 10 to 70.439791, 20 fc rgb "#0000FF"
set object 27 rect from 0.084176, 10 to 71.149511, 20 fc rgb "#FFFF00"
set object 28 rect from 0.085072, 10 to 72.151664, 20 fc rgb "#FF00FF"
set object 29 rect from 0.086268, 10 to 78.790508, 20 fc rgb "#808080"
set object 30 rect from 0.094166, 10 to 79.292423, 20 fc rgb "#800080"
set object 31 rect from 0.095020, 10 to 79.806069, 20 fc rgb "#008080"
set object 32 rect from 0.095438, 10 to 148.250771, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.175875, 20 to 160.136676, 30 fc rgb "#FF0000"
set object 34 rect from 0.054565, 20 to 47.468199, 30 fc rgb "#00FF00"
set object 35 rect from 0.056996, 20 to 48.059771, 30 fc rgb "#0000FF"
set object 36 rect from 0.057469, 20 to 50.686653, 30 fc rgb "#FFFF00"
set object 37 rect from 0.060611, 20 to 53.963761, 30 fc rgb "#FF00FF"
set object 38 rect from 0.064552, 20 to 60.291736, 30 fc rgb "#808080"
set object 39 rect from 0.072099, 20 to 60.874929, 30 fc rgb "#800080"
set object 40 rect from 0.073050, 20 to 61.424604, 30 fc rgb "#008080"
set object 41 rect from 0.073439, 20 to 146.855633, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.174026, 30 to 161.543542, 40 fc rgb "#FF0000"
set object 43 rect from 0.020219, 30 to 19.494222, 40 fc rgb "#00FF00"
set object 44 rect from 0.023715, 30 to 20.824002, 40 fc rgb "#0000FF"
set object 45 rect from 0.024993, 30 to 25.144321, 40 fc rgb "#FFFF00"
set object 46 rect from 0.030179, 30 to 28.713024, 40 fc rgb "#FF00FF"
set object 47 rect from 0.034394, 30 to 35.242939, 40 fc rgb "#808080"
set object 48 rect from 0.042271, 30 to 36.004608, 40 fc rgb "#800080"
set object 49 rect from 0.043549, 30 to 37.101448, 40 fc rgb "#008080"
set object 50 rect from 0.044405, 30 to 144.892385, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.182441, 40 to 164.756131, 50 fc rgb "#FF0000"
set object 52 rect from 0.150174, 40 to 127.252482, 50 fc rgb "#00FF00"
set object 53 rect from 0.152307, 40 to 127.854946, 50 fc rgb "#0000FF"
set object 54 rect from 0.152698, 40 to 130.404738, 50 fc rgb "#FFFF00"
set object 55 rect from 0.155745, 40 to 132.999779, 50 fc rgb "#FF00FF"
set object 56 rect from 0.158839, 40 to 139.270777, 50 fc rgb "#808080"
set object 57 rect from 0.166325, 40 to 139.796153, 50 fc rgb "#800080"
set object 58 rect from 0.167216, 40 to 140.323204, 50 fc rgb "#008080"
set object 59 rect from 0.167584, 40 to 152.481431, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.180996, 50 to 164.410906, 60 fc rgb "#FF0000"
set object 61 rect from 0.123893, 50 to 105.262933, 60 fc rgb "#00FF00"
set object 62 rect from 0.125965, 50 to 105.927403, 60 fc rgb "#0000FF"
set object 63 rect from 0.126542, 50 to 108.610425, 60 fc rgb "#FFFF00"
set object 64 rect from 0.129745, 50 to 111.682242, 60 fc rgb "#FF00FF"
set object 65 rect from 0.133418, 50 to 118.036194, 60 fc rgb "#808080"
set object 66 rect from 0.141005, 50 to 118.641173, 60 fc rgb "#800080"
set object 67 rect from 0.142007, 50 to 119.217662, 60 fc rgb "#008080"
set object 68 rect from 0.142397, 50 to 151.130703, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
