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

set object 15 rect from 0.138761, 0 to 151.929111, 10 fc rgb "#FF0000"
set object 16 rect from 0.048734, 0 to 54.835459, 10 fc rgb "#00FF00"
set object 17 rect from 0.052507, 0 to 56.444824, 10 fc rgb "#0000FF"
set object 18 rect from 0.053783, 0 to 58.229865, 10 fc rgb "#FFFF00"
set object 19 rect from 0.055498, 0 to 59.552078, 10 fc rgb "#FF00FF"
set object 20 rect from 0.056726, 0 to 60.968964, 10 fc rgb "#808080"
set object 21 rect from 0.058090, 0 to 61.608503, 10 fc rgb "#800080"
set object 22 rect from 0.059034, 0 to 62.313246, 10 fc rgb "#008080"
set object 23 rect from 0.059354, 0 to 145.151803, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.142057, 10 to 152.401343, 20 fc rgb "#FF0000"
set object 25 rect from 0.084981, 10 to 91.025320, 20 fc rgb "#00FF00"
set object 26 rect from 0.086841, 10 to 91.635392, 20 fc rgb "#0000FF"
set object 27 rect from 0.087215, 10 to 92.225494, 20 fc rgb "#FFFF00"
set object 28 rect from 0.087796, 10 to 93.197452, 20 fc rgb "#FF00FF"
set object 29 rect from 0.088716, 10 to 94.169378, 20 fc rgb "#808080"
set object 30 rect from 0.089636, 10 to 94.681643, 20 fc rgb "#800080"
set object 31 rect from 0.090353, 10 to 95.202340, 20 fc rgb "#008080"
set object 32 rect from 0.090610, 10 to 148.733454, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.140495, 20 to 151.627194, 30 fc rgb "#FF0000"
set object 34 rect from 0.069486, 20 to 74.887408, 30 fc rgb "#00FF00"
set object 35 rect from 0.071529, 20 to 75.627920, 30 fc rgb "#0000FF"
set object 36 rect from 0.071999, 20 to 76.634581, 30 fc rgb "#FFFF00"
set object 37 rect from 0.072993, 20 to 78.009365, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074287, 20 to 78.915052, 30 fc rgb "#808080"
set object 39 rect from 0.075154, 20 to 79.495688, 30 fc rgb "#800080"
set object 40 rect from 0.075931, 20 to 80.649591, 30 fc rgb "#008080"
set object 41 rect from 0.076800, 20 to 147.197726, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.143634, 30 to 154.166416, 40 fc rgb "#FF0000"
set object 43 rect from 0.098574, 30 to 104.961665, 40 fc rgb "#00FF00"
set object 44 rect from 0.100125, 30 to 105.585404, 40 fc rgb "#0000FF"
set object 45 rect from 0.100481, 30 to 106.318581, 40 fc rgb "#FFFF00"
set object 46 rect from 0.101189, 30 to 107.343110, 40 fc rgb "#FF00FF"
set object 47 rect from 0.102160, 30 to 108.206696, 40 fc rgb "#808080"
set object 48 rect from 0.102987, 30 to 108.761030, 40 fc rgb "#800080"
set object 49 rect from 0.103753, 30 to 109.491040, 40 fc rgb "#008080"
set object 50 rect from 0.104202, 30 to 150.426989, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.146359, 40 to 156.775105, 50 fc rgb "#FF0000"
set object 52 rect from 0.126924, 40 to 134.660367, 50 fc rgb "#00FF00"
set object 53 rect from 0.128342, 40 to 135.227335, 50 fc rgb "#0000FF"
set object 54 rect from 0.128659, 40 to 135.845839, 50 fc rgb "#FFFF00"
set object 55 rect from 0.129247, 40 to 136.849333, 50 fc rgb "#FF00FF"
set object 56 rect from 0.130204, 40 to 137.669815, 50 fc rgb "#808080"
set object 57 rect from 0.130984, 40 to 138.144211, 50 fc rgb "#800080"
set object 58 rect from 0.131670, 40 to 138.739548, 50 fc rgb "#008080"
set object 59 rect from 0.132013, 40 to 153.485842, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.145019, 50 to 155.670607, 60 fc rgb "#FF0000"
set object 61 rect from 0.112578, 50 to 119.618521, 60 fc rgb "#00FF00"
set object 62 rect from 0.114022, 50 to 120.179188, 60 fc rgb "#0000FF"
set object 63 rect from 0.114351, 50 to 120.909167, 60 fc rgb "#FFFF00"
set object 64 rect from 0.115077, 50 to 122.092538, 60 fc rgb "#FF00FF"
set object 65 rect from 0.116172, 50 to 122.925622, 60 fc rgb "#808080"
set object 66 rect from 0.116966, 50 to 123.404218, 60 fc rgb "#800080"
set object 67 rect from 0.117635, 50 to 123.986985, 60 fc rgb "#008080"
set object 68 rect from 0.117972, 50 to 152.026886, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
