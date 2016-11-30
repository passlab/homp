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

set object 15 rect from 0.153935, 0 to 158.788539, 10 fc rgb "#FF0000"
set object 16 rect from 0.131959, 0 to 133.179479, 10 fc rgb "#00FF00"
set object 17 rect from 0.133983, 0 to 133.868218, 10 fc rgb "#0000FF"
set object 18 rect from 0.134419, 0 to 134.518088, 10 fc rgb "#FFFF00"
set object 19 rect from 0.135074, 0 to 135.507842, 10 fc rgb "#FF00FF"
set object 20 rect from 0.136080, 0 to 138.697383, 10 fc rgb "#808080"
set object 21 rect from 0.139282, 0 to 139.271505, 10 fc rgb "#800080"
set object 22 rect from 0.140107, 0 to 139.800773, 10 fc rgb "#008080"
set object 23 rect from 0.140383, 0 to 152.955659, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.150984, 10 to 155.928917, 20 fc rgb "#FF0000"
set object 25 rect from 0.098523, 10 to 99.695218, 20 fc rgb "#00FF00"
set object 26 rect from 0.100372, 10 to 100.352069, 20 fc rgb "#0000FF"
set object 27 rect from 0.100791, 10 to 101.054769, 20 fc rgb "#FFFF00"
set object 28 rect from 0.101524, 10 to 102.160150, 20 fc rgb "#FF00FF"
set object 29 rect from 0.102642, 10 to 105.300855, 20 fc rgb "#808080"
set object 30 rect from 0.105817, 10 to 105.932784, 20 fc rgb "#800080"
set object 31 rect from 0.106642, 10 to 106.488950, 20 fc rgb "#008080"
set object 32 rect from 0.106962, 10 to 149.878730, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.149440, 20 to 156.181083, 30 fc rgb "#FF0000"
set object 34 rect from 0.080204, 20 to 81.319448, 30 fc rgb "#00FF00"
set object 35 rect from 0.081944, 20 to 82.146746, 30 fc rgb "#0000FF"
set object 36 rect from 0.082527, 20 to 84.023594, 30 fc rgb "#FFFF00"
set object 37 rect from 0.084410, 20 to 85.724010, 30 fc rgb "#FF00FF"
set object 38 rect from 0.086125, 20 to 88.670356, 30 fc rgb "#808080"
set object 39 rect from 0.089078, 20 to 89.250464, 30 fc rgb "#800080"
set object 40 rect from 0.089900, 20 to 89.831552, 30 fc rgb "#008080"
set object 41 rect from 0.090246, 20 to 148.498266, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.148061, 30 to 154.932167, 40 fc rgb "#FF0000"
set object 43 rect from 0.060665, 30 to 62.212075, 40 fc rgb "#00FF00"
set object 44 rect from 0.062785, 30 to 62.900829, 40 fc rgb "#0000FF"
set object 45 rect from 0.063217, 30 to 64.818537, 40 fc rgb "#FFFF00"
set object 46 rect from 0.065163, 30 to 66.739244, 40 fc rgb "#FF00FF"
set object 47 rect from 0.067078, 30 to 69.720479, 40 fc rgb "#808080"
set object 48 rect from 0.070090, 30 to 70.326489, 40 fc rgb "#800080"
set object 49 rect from 0.070945, 30 to 71.011248, 40 fc rgb "#008080"
set object 50 rect from 0.071372, 30 to 146.951336, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.146372, 40 to 156.147204, 50 fc rgb "#FF0000"
set object 52 rect from 0.035638, 40 to 38.646329, 50 fc rgb "#00FF00"
set object 53 rect from 0.039215, 40 to 40.003876, 50 fc rgb "#0000FF"
set object 54 rect from 0.040254, 40 to 43.411703, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043705, 40 to 45.798882, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046074, 40 to 49.065180, 50 fc rgb "#808080"
set object 57 rect from 0.049371, 40 to 49.752924, 50 fc rgb "#800080"
set object 58 rect from 0.050455, 40 to 50.940024, 50 fc rgb "#008080"
set object 59 rect from 0.051236, 40 to 145.039599, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.152448, 50 to 158.810447, 60 fc rgb "#FF0000"
set object 61 rect from 0.114220, 50 to 115.218346, 60 fc rgb "#00FF00"
set object 62 rect from 0.115988, 50 to 115.874187, 60 fc rgb "#0000FF"
set object 63 rect from 0.116365, 50 to 117.706195, 60 fc rgb "#FFFF00"
set object 64 rect from 0.118204, 50 to 119.287999, 60 fc rgb "#FF00FF"
set object 65 rect from 0.119799, 50 to 122.222389, 60 fc rgb "#808080"
set object 66 rect from 0.122745, 50 to 122.821419, 60 fc rgb "#800080"
set object 67 rect from 0.123582, 50 to 123.429435, 60 fc rgb "#008080"
set object 68 rect from 0.123961, 50 to 151.396757, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
