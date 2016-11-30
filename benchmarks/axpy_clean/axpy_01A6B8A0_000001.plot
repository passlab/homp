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

set object 15 rect from 0.118034, 0 to 162.631096, 10 fc rgb "#FF0000"
set object 16 rect from 0.097683, 0 to 132.384922, 10 fc rgb "#00FF00"
set object 17 rect from 0.099559, 0 to 133.166995, 10 fc rgb "#0000FF"
set object 18 rect from 0.099920, 0 to 134.063858, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100589, 0 to 135.445171, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101617, 0 to 137.806088, 10 fc rgb "#808080"
set object 21 rect from 0.103394, 0 to 138.540124, 10 fc rgb "#800080"
set object 22 rect from 0.104197, 0 to 139.251468, 10 fc rgb "#008080"
set object 23 rect from 0.104462, 0 to 156.698770, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.116409, 10 to 160.940139, 20 fc rgb "#FF0000"
set object 25 rect from 0.082976, 10 to 112.874316, 20 fc rgb "#00FF00"
set object 26 rect from 0.084937, 10 to 113.759156, 20 fc rgb "#0000FF"
set object 27 rect from 0.085362, 10 to 114.741423, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086123, 10 to 116.288238, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087281, 10 to 118.731896, 20 fc rgb "#808080"
set object 30 rect from 0.089114, 10 to 119.538003, 20 fc rgb "#800080"
set object 31 rect from 0.089963, 10 to 120.336095, 20 fc rgb "#008080"
set object 32 rect from 0.090289, 10 to 154.515357, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.114702, 20 to 159.791058, 30 fc rgb "#FF0000"
set object 34 rect from 0.067737, 20 to 92.358750, 30 fc rgb "#00FF00"
set object 35 rect from 0.069573, 20 to 93.146163, 30 fc rgb "#0000FF"
set object 36 rect from 0.069913, 20 to 95.673913, 30 fc rgb "#FFFF00"
set object 37 rect from 0.071826, 20 to 97.202034, 30 fc rgb "#FF00FF"
set object 38 rect from 0.072955, 20 to 99.182589, 30 fc rgb "#808080"
set object 39 rect from 0.074449, 20 to 100.091453, 30 fc rgb "#800080"
set object 40 rect from 0.075396, 20 to 100.922916, 30 fc rgb "#008080"
set object 41 rect from 0.075740, 20 to 152.425363, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.113181, 30 to 157.857208, 40 fc rgb "#FF0000"
set object 43 rect from 0.052113, 30 to 71.672354, 40 fc rgb "#00FF00"
set object 44 rect from 0.054079, 30 to 72.489130, 40 fc rgb "#0000FF"
set object 45 rect from 0.054434, 30 to 75.020878, 40 fc rgb "#FFFF00"
set object 46 rect from 0.056346, 30 to 76.573032, 40 fc rgb "#FF00FF"
set object 47 rect from 0.057495, 30 to 78.601634, 40 fc rgb "#808080"
set object 48 rect from 0.059026, 30 to 79.495811, 40 fc rgb "#800080"
set object 49 rect from 0.059993, 30 to 80.399346, 40 fc rgb "#008080"
set object 50 rect from 0.060380, 30 to 150.370073, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.111648, 40 to 156.012792, 50 fc rgb "#FF0000"
set object 52 rect from 0.035950, 40 to 50.408076, 50 fc rgb "#00FF00"
set object 53 rect from 0.038149, 40 to 51.272900, 50 fc rgb "#0000FF"
set object 54 rect from 0.038539, 40 to 53.808654, 50 fc rgb "#FFFF00"
set object 55 rect from 0.040457, 40 to 55.524958, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041741, 40 to 57.532201, 50 fc rgb "#808080"
set object 57 rect from 0.043241, 40 to 58.441075, 50 fc rgb "#800080"
set object 58 rect from 0.044175, 40 to 59.297884, 50 fc rgb "#008080"
set object 59 rect from 0.044573, 40 to 148.236039, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.109929, 50 to 156.545291, 60 fc rgb "#FF0000"
set object 61 rect from 0.015953, 50 to 24.747607, 60 fc rgb "#00FF00"
set object 62 rect from 0.019089, 50 to 26.327783, 60 fc rgb "#0000FF"
set object 63 rect from 0.019866, 50 to 30.247516, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022828, 50 to 32.529690, 60 fc rgb "#FF00FF"
set object 65 rect from 0.024515, 50 to 34.873264, 60 fc rgb "#808080"
set object 66 rect from 0.026283, 50 to 35.927603, 60 fc rgb "#800080"
set object 67 rect from 0.027489, 50 to 37.491760, 60 fc rgb "#008080"
set object 68 rect from 0.028229, 50 to 145.319922, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
