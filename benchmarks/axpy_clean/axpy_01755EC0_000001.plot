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

set object 15 rect from 0.271906, 0 to 156.444187, 10 fc rgb "#FF0000"
set object 16 rect from 0.078852, 0 to 43.449484, 10 fc rgb "#00FF00"
set object 17 rect from 0.081128, 0 to 43.869507, 10 fc rgb "#0000FF"
set object 18 rect from 0.081687, 0 to 44.238441, 10 fc rgb "#FFFF00"
set object 19 rect from 0.082409, 0 to 44.846693, 10 fc rgb "#FF00FF"
set object 20 rect from 0.083542, 0 to 53.760205, 10 fc rgb "#808080"
set object 21 rect from 0.100098, 0 to 54.083965, 10 fc rgb "#800080"
set object 22 rect from 0.100949, 0 to 54.410947, 10 fc rgb "#008080"
set object 23 rect from 0.101313, 0 to 145.802173, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.273470, 10 to 157.001335, 20 fc rgb "#FF0000"
set object 25 rect from 0.109411, 10 to 59.646442, 20 fc rgb "#00FF00"
set object 26 rect from 0.111231, 10 to 59.992248, 20 fc rgb "#0000FF"
set object 27 rect from 0.111658, 10 to 60.306319, 20 fc rgb "#FFFF00"
set object 28 rect from 0.112254, 10 to 60.811313, 20 fc rgb "#FF00FF"
set object 29 rect from 0.113189, 10 to 69.661371, 20 fc rgb "#808080"
set object 30 rect from 0.129639, 10 to 69.933498, 20 fc rgb "#800080"
set object 31 rect from 0.130400, 10 to 70.210465, 20 fc rgb "#008080"
set object 32 rect from 0.130682, 10 to 146.749772, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.269763, 20 to 163.661462, 30 fc rgb "#FF0000"
set object 34 rect from 0.028845, 20 to 17.476915, 30 fc rgb "#00FF00"
set object 35 rect from 0.033025, 20 to 18.052367, 30 fc rgb "#0000FF"
set object 36 rect from 0.033684, 20 to 20.109989, 30 fc rgb "#FFFF00"
set object 37 rect from 0.037550, 20 to 26.978789, 30 fc rgb "#FF00FF"
set object 38 rect from 0.050284, 20 to 35.960067, 30 fc rgb "#808080"
set object 39 rect from 0.067022, 20 to 36.579074, 30 fc rgb "#800080"
set object 40 rect from 0.068579, 20 to 37.223901, 30 fc rgb "#008080"
set object 41 rect from 0.069331, 20 to 144.563078, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.275224, 30 to 167.592813, 40 fc rgb "#FF0000"
set object 43 rect from 0.138573, 30 to 75.309893, 40 fc rgb "#00FF00"
set object 44 rect from 0.140357, 30 to 75.607840, 40 fc rgb "#0000FF"
set object 45 rect from 0.140699, 30 to 79.758587, 40 fc rgb "#FFFF00"
set object 46 rect from 0.148486, 30 to 85.951362, 40 fc rgb "#FF00FF"
set object 47 rect from 0.159956, 30 to 94.742262, 40 fc rgb "#808080"
set object 48 rect from 0.176298, 30 to 95.304266, 40 fc rgb "#800080"
set object 49 rect from 0.177576, 30 to 95.668352, 40 fc rgb "#008080"
set object 50 rect from 0.178008, 30 to 147.719433, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.276555, 40 to 165.805137, 50 fc rgb "#FF0000"
set object 52 rect from 0.185941, 40 to 100.746807, 50 fc rgb "#00FF00"
set object 53 rect from 0.187670, 40 to 101.052280, 50 fc rgb "#0000FF"
set object 54 rect from 0.188006, 40 to 102.607602, 50 fc rgb "#FFFF00"
set object 55 rect from 0.190899, 40 to 108.868141, 50 fc rgb "#FF00FF"
set object 56 rect from 0.202549, 40 to 117.660116, 50 fc rgb "#808080"
set object 57 rect from 0.218891, 40 to 118.183397, 50 fc rgb "#800080"
set object 58 rect from 0.220126, 40 to 118.508767, 50 fc rgb "#008080"
set object 59 rect from 0.220477, 40 to 148.486335, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.277860, 50 to 166.292933, 60 fc rgb "#FF0000"
set object 61 rect from 0.229441, 50 to 124.088450, 60 fc rgb "#00FF00"
set object 62 rect from 0.231054, 50 to 124.432108, 60 fc rgb "#0000FF"
set object 63 rect from 0.231479, 50 to 125.890086, 60 fc rgb "#FFFF00"
set object 64 rect from 0.234218, 50 to 131.979610, 60 fc rgb "#FF00FF"
set object 65 rect from 0.245524, 50 to 140.769429, 60 fc rgb "#808080"
set object 66 rect from 0.261858, 50 to 141.314756, 60 fc rgb "#800080"
set object 67 rect from 0.263118, 50 to 141.626679, 60 fc rgb "#008080"
set object 68 rect from 0.263454, 50 to 149.195153, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
