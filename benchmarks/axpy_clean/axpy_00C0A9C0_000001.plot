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

set object 15 rect from 0.176038, 0 to 156.250956, 10 fc rgb "#FF0000"
set object 16 rect from 0.135480, 0 to 119.416621, 10 fc rgb "#00FF00"
set object 17 rect from 0.138089, 0 to 120.087025, 10 fc rgb "#0000FF"
set object 18 rect from 0.138590, 0 to 120.858875, 10 fc rgb "#FFFF00"
set object 19 rect from 0.139505, 0 to 121.903070, 10 fc rgb "#FF00FF"
set object 20 rect from 0.140695, 0 to 123.161453, 10 fc rgb "#808080"
set object 21 rect from 0.142144, 0 to 123.700040, 10 fc rgb "#800080"
set object 22 rect from 0.143050, 0 to 124.235138, 10 fc rgb "#008080"
set object 23 rect from 0.143381, 0 to 152.113233, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.168608, 10 to 152.349986, 20 fc rgb "#FF0000"
set object 25 rect from 0.052217, 10 to 49.315917, 20 fc rgb "#00FF00"
set object 26 rect from 0.057355, 10 to 50.720022, 20 fc rgb "#0000FF"
set object 27 rect from 0.058639, 10 to 52.444131, 20 fc rgb "#FFFF00"
set object 28 rect from 0.060656, 10 to 54.103211, 20 fc rgb "#FF00FF"
set object 29 rect from 0.062533, 10 to 55.680745, 20 fc rgb "#808080"
set object 30 rect from 0.064356, 10 to 56.306926, 20 fc rgb "#800080"
set object 31 rect from 0.065603, 10 to 57.081387, 20 fc rgb "#008080"
set object 32 rect from 0.065967, 10 to 145.362467, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.177645, 20 to 157.596053, 30 fc rgb "#FF0000"
set object 34 rect from 0.152568, 20 to 134.092407, 30 fc rgb "#00FF00"
set object 35 rect from 0.155012, 20 to 134.713367, 30 fc rgb "#0000FF"
set object 36 rect from 0.155464, 20 to 135.534662, 30 fc rgb "#FFFF00"
set object 37 rect from 0.156434, 20 to 136.666449, 30 fc rgb "#FF00FF"
set object 38 rect from 0.157713, 20 to 137.783478, 30 fc rgb "#808080"
set object 39 rect from 0.159039, 20 to 138.363678, 30 fc rgb "#800080"
set object 40 rect from 0.159941, 20 to 138.910045, 30 fc rgb "#008080"
set object 41 rect from 0.160291, 20 to 153.617053, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.170671, 30 to 152.722045, 40 fc rgb "#FF0000"
set object 43 rect from 0.077453, 30 to 69.227380, 40 fc rgb "#00FF00"
set object 44 rect from 0.080354, 30 to 70.123242, 40 fc rgb "#0000FF"
set object 45 rect from 0.080980, 30 to 71.072890, 40 fc rgb "#FFFF00"
set object 46 rect from 0.082112, 30 to 72.934916, 40 fc rgb "#FF00FF"
set object 47 rect from 0.084245, 30 to 74.120464, 40 fc rgb "#808080"
set object 48 rect from 0.085611, 30 to 74.789988, 40 fc rgb "#800080"
set object 49 rect from 0.086703, 30 to 75.811619, 40 fc rgb "#008080"
set object 50 rect from 0.087572, 30 to 147.364967, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.172467, 40 to 153.196403, 50 fc rgb "#FF0000"
set object 52 rect from 0.097243, 40 to 86.027946, 50 fc rgb "#00FF00"
set object 53 rect from 0.099605, 40 to 86.717425, 50 fc rgb "#0000FF"
set object 54 rect from 0.100114, 40 to 87.536109, 50 fc rgb "#FFFF00"
set object 55 rect from 0.101059, 40 to 88.636674, 50 fc rgb "#FF00FF"
set object 56 rect from 0.102327, 40 to 89.757167, 50 fc rgb "#808080"
set object 57 rect from 0.103642, 40 to 90.348635, 50 fc rgb "#800080"
set object 58 rect from 0.104570, 40 to 90.940983, 50 fc rgb "#008080"
set object 59 rect from 0.105002, 40 to 149.040511, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.174225, 50 to 154.535453, 60 fc rgb "#FF0000"
set object 61 rect from 0.114604, 50 to 101.018542, 60 fc rgb "#00FF00"
set object 62 rect from 0.116863, 50 to 101.626500, 60 fc rgb "#0000FF"
set object 63 rect from 0.117325, 50 to 102.437379, 60 fc rgb "#FFFF00"
set object 64 rect from 0.118240, 50 to 103.453814, 60 fc rgb "#FF00FF"
set object 65 rect from 0.119408, 50 to 104.567380, 60 fc rgb "#808080"
set object 66 rect from 0.120703, 50 to 105.131115, 60 fc rgb "#800080"
set object 67 rect from 0.121619, 50 to 105.713900, 60 fc rgb "#008080"
set object 68 rect from 0.122020, 50 to 150.610266, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
