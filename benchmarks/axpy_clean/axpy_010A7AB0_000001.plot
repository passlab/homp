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

set object 15 rect from 0.124005, 0 to 160.641335, 10 fc rgb "#FF0000"
set object 16 rect from 0.103567, 0 to 133.686567, 10 fc rgb "#00FF00"
set object 17 rect from 0.105697, 0 to 134.491017, 10 fc rgb "#0000FF"
set object 18 rect from 0.106105, 0 to 135.413527, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106831, 0 to 136.703945, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107867, 0 to 137.231812, 10 fc rgb "#808080"
set object 21 rect from 0.108267, 0 to 137.864995, 10 fc rgb "#800080"
set object 22 rect from 0.109016, 0 to 138.542574, 10 fc rgb "#008080"
set object 23 rect from 0.109303, 0 to 156.592387, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.122359, 10 to 158.942358, 20 fc rgb "#FF0000"
set object 25 rect from 0.089191, 10 to 115.579653, 20 fc rgb "#00FF00"
set object 26 rect from 0.091439, 10 to 116.406981, 20 fc rgb "#0000FF"
set object 27 rect from 0.091857, 10 to 117.408110, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092672, 10 to 118.855916, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093791, 10 to 119.459906, 20 fc rgb "#808080"
set object 30 rect from 0.094279, 10 to 120.186947, 20 fc rgb "#800080"
set object 31 rect from 0.095105, 10 to 120.931800, 20 fc rgb "#008080"
set object 32 rect from 0.095425, 10 to 154.486023, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.120724, 20 to 156.715439, 30 fc rgb "#FF0000"
set object 34 rect from 0.075790, 20 to 98.409203, 30 fc rgb "#00FF00"
set object 35 rect from 0.077931, 20 to 99.242847, 30 fc rgb "#0000FF"
set object 36 rect from 0.078346, 20 to 100.336623, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079209, 20 to 101.704449, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080283, 20 to 102.233602, 30 fc rgb "#808080"
set object 39 rect from 0.080685, 20 to 102.925135, 30 fc rgb "#800080"
set object 40 rect from 0.081484, 20 to 103.700429, 30 fc rgb "#008080"
set object 41 rect from 0.081842, 20 to 152.564956, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119098, 30 to 154.761396, 40 fc rgb "#FF0000"
set object 43 rect from 0.060164, 30 to 78.674330, 40 fc rgb "#00FF00"
set object 44 rect from 0.062396, 30 to 79.537167, 40 fc rgb "#0000FF"
set object 45 rect from 0.062796, 30 to 80.637259, 40 fc rgb "#FFFF00"
set object 46 rect from 0.063664, 30 to 82.007657, 40 fc rgb "#FF00FF"
set object 47 rect from 0.064743, 30 to 82.572318, 40 fc rgb "#808080"
set object 48 rect from 0.065207, 30 to 83.268918, 40 fc rgb "#800080"
set object 49 rect from 0.065994, 30 to 84.002351, 40 fc rgb "#008080"
set object 50 rect from 0.066316, 30 to 150.476366, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.117524, 40 to 152.973591, 50 fc rgb "#FF0000"
set object 52 rect from 0.044369, 40 to 58.916578, 50 fc rgb "#00FF00"
set object 53 rect from 0.046821, 40 to 59.787054, 50 fc rgb "#0000FF"
set object 54 rect from 0.047253, 40 to 60.926512, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048158, 40 to 62.455508, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049370, 40 to 63.119133, 50 fc rgb "#808080"
set object 57 rect from 0.049860, 40 to 63.806847, 50 fc rgb "#800080"
set object 58 rect from 0.050696, 40 to 64.617687, 50 fc rgb "#008080"
set object 59 rect from 0.051055, 40 to 148.310405, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.115603, 50 to 153.750095, 60 fc rgb "#FF0000"
set object 61 rect from 0.023701, 50 to 33.722994, 60 fc rgb "#00FF00"
set object 62 rect from 0.027115, 50 to 35.358592, 60 fc rgb "#0000FF"
set object 63 rect from 0.027998, 50 to 38.053705, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030146, 50 to 40.153678, 60 fc rgb "#FF00FF"
set object 65 rect from 0.031777, 50 to 41.090104, 60 fc rgb "#808080"
set object 66 rect from 0.032607, 50 to 42.117893, 60 fc rgb "#800080"
set object 67 rect from 0.033783, 50 to 43.399462, 60 fc rgb "#008080"
set object 68 rect from 0.034359, 50 to 145.361511, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
