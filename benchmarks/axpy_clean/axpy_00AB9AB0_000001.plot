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

set object 15 rect from 0.121145, 0 to 161.486469, 10 fc rgb "#FF0000"
set object 16 rect from 0.100373, 0 to 133.434275, 10 fc rgb "#00FF00"
set object 17 rect from 0.102668, 0 to 134.215459, 10 fc rgb "#0000FF"
set object 18 rect from 0.103037, 0 to 135.187003, 10 fc rgb "#FFFF00"
set object 19 rect from 0.103777, 0 to 136.552404, 10 fc rgb "#FF00FF"
set object 20 rect from 0.104825, 0 to 137.107938, 10 fc rgb "#808080"
set object 21 rect from 0.105272, 0 to 137.864248, 10 fc rgb "#800080"
set object 22 rect from 0.106094, 0 to 138.580217, 10 fc rgb "#008080"
set object 23 rect from 0.106386, 0 to 157.287353, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.119584, 10 to 159.985585, 20 fc rgb "#FF0000"
set object 25 rect from 0.086117, 10 to 114.762351, 20 fc rgb "#00FF00"
set object 26 rect from 0.088440, 10 to 115.823905, 20 fc rgb "#0000FF"
set object 27 rect from 0.088932, 10 to 116.884139, 20 fc rgb "#FFFF00"
set object 28 rect from 0.089775, 10 to 118.505115, 20 fc rgb "#FF00FF"
set object 29 rect from 0.091015, 10 to 119.137602, 20 fc rgb "#808080"
set object 30 rect from 0.091491, 10 to 119.920029, 20 fc rgb "#800080"
set object 31 rect from 0.092330, 10 to 120.688154, 20 fc rgb "#008080"
set object 32 rect from 0.092679, 10 to 155.113486, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.117851, 20 to 157.394387, 30 fc rgb "#FF0000"
set object 34 rect from 0.072972, 20 to 97.534022, 30 fc rgb "#00FF00"
set object 35 rect from 0.075145, 20 to 98.432578, 30 fc rgb "#0000FF"
set object 36 rect from 0.075594, 20 to 99.521494, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076432, 20 to 100.933765, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077512, 20 to 101.477640, 30 fc rgb "#808080"
set object 39 rect from 0.077936, 20 to 102.159641, 30 fc rgb "#800080"
set object 40 rect from 0.078712, 20 to 102.956448, 30 fc rgb "#008080"
set object 41 rect from 0.079065, 20 to 152.963015, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.116122, 30 to 155.238631, 40 fc rgb "#FF0000"
set object 43 rect from 0.058211, 30 to 78.341774, 40 fc rgb "#00FF00"
set object 44 rect from 0.060509, 30 to 79.244217, 40 fc rgb "#0000FF"
set object 45 rect from 0.060879, 30 to 80.449183, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061807, 30 to 81.865419, 40 fc rgb "#FF00FF"
set object 47 rect from 0.062892, 30 to 82.449635, 40 fc rgb "#808080"
set object 48 rect from 0.063337, 30 to 83.152545, 40 fc rgb "#800080"
set object 49 rect from 0.064140, 30 to 83.905047, 40 fc rgb "#008080"
set object 50 rect from 0.064480, 30 to 150.763031, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.114497, 40 to 153.128580, 50 fc rgb "#FF0000"
set object 52 rect from 0.044087, 40 to 60.320603, 50 fc rgb "#00FF00"
set object 53 rect from 0.046646, 40 to 61.237347, 50 fc rgb "#0000FF"
set object 54 rect from 0.047072, 40 to 62.327584, 50 fc rgb "#FFFF00"
set object 55 rect from 0.047910, 40 to 63.807714, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049065, 40 to 64.415405, 50 fc rgb "#808080"
set object 57 rect from 0.049524, 40 to 65.122201, 50 fc rgb "#800080"
set object 58 rect from 0.050326, 40 to 65.892969, 50 fc rgb "#008080"
set object 59 rect from 0.050659, 40 to 148.457411, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.112622, 50 to 155.519002, 60 fc rgb "#FF0000"
set object 61 rect from 0.023056, 50 to 33.975354, 60 fc rgb "#00FF00"
set object 62 rect from 0.026576, 50 to 35.644600, 60 fc rgb "#0000FF"
set object 63 rect from 0.027476, 50 to 39.413404, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030386, 50 to 41.875571, 60 fc rgb "#FF00FF"
set object 65 rect from 0.032245, 50 to 42.767520, 60 fc rgb "#808080"
set object 66 rect from 0.032935, 50 to 43.876023, 60 fc rgb "#800080"
set object 67 rect from 0.034285, 50 to 45.237459, 60 fc rgb "#008080"
set object 68 rect from 0.034826, 50 to 145.421444, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
