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

set object 15 rect from 1.394224, 0 to 272.234932, 10 fc rgb "#FF0000"
set object 16 rect from 0.155116, 0 to 16.373941, 10 fc rgb "#00FF00"
set object 17 rect from 0.156986, 0 to 16.438854, 10 fc rgb "#0000FF"
set object 18 rect from 0.157379, 0 to 141.431755, 10 fc rgb "#FFFF00"
set object 19 rect from 1.353625, 0 to 141.857078, 10 fc rgb "#FF00FF"
set object 20 rect from 1.357275, 0 to 142.934236, 10 fc rgb "#808080"
set object 21 rect from 1.367622, 0 to 143.004689, 10 fc rgb "#800080"
set object 22 rect from 1.368593, 0 to 143.079320, 10 fc rgb "#008080"
set object 23 rect from 1.368949, 0 to 145.663243, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.392514, 10 to 146.826741, 20 fc rgb "#FF0000"
set object 25 rect from 0.132245, 10 to 13.980677, 20 fc rgb "#00FF00"
set object 26 rect from 0.134094, 10 to 14.049143, 20 fc rgb "#0000FF"
set object 27 rect from 0.134536, 10 to 14.139246, 20 fc rgb "#FFFF00"
set object 28 rect from 0.135419, 10 to 14.256317, 20 fc rgb "#FF00FF"
set object 29 rect from 0.136536, 10 to 15.273998, 20 fc rgb "#808080"
set object 30 rect from 0.146249, 10 to 15.333370, 20 fc rgb "#800080"
set object 31 rect from 0.147078, 10 to 15.396609, 20 fc rgb "#008080"
set object 32 rect from 0.147417, 10 to 145.451576, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.389524, 20 to 147.104992, 30 fc rgb "#FF0000"
set object 34 rect from 0.105572, 20 to 11.185400, 30 fc rgb "#00FF00"
set object 35 rect from 0.107367, 20 to 11.244352, 30 fc rgb "#0000FF"
set object 36 rect from 0.107701, 20 to 11.562744, 30 fc rgb "#FFFF00"
set object 37 rect from 0.110756, 20 to 12.049738, 30 fc rgb "#FF00FF"
set object 38 rect from 0.115391, 20 to 13.048604, 30 fc rgb "#808080"
set object 39 rect from 0.124945, 20 to 13.123655, 30 fc rgb "#800080"
set object 40 rect from 0.125925, 20 to 13.186581, 30 fc rgb "#008080"
set object 41 rect from 0.126278, 20 to 145.157330, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.386684, 30 to 146.823395, 40 fc rgb "#FF0000"
set object 43 rect from 0.076688, 30 to 8.161833, 40 fc rgb "#00FF00"
set object 44 rect from 0.078426, 30 to 8.223921, 40 fc rgb "#0000FF"
set object 45 rect from 0.078789, 30 to 8.532801, 40 fc rgb "#FFFF00"
set object 46 rect from 0.081746, 30 to 9.031712, 40 fc rgb "#FF00FF"
set object 47 rect from 0.086525, 30 to 10.035803, 40 fc rgb "#808080"
set object 48 rect from 0.096137, 30 to 10.112005, 40 fc rgb "#800080"
set object 49 rect from 0.097113, 30 to 10.171586, 40 fc rgb "#008080"
set object 50 rect from 0.097422, 30 to 144.853573, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.384003, 40 to 146.596364, 50 fc rgb "#FF0000"
set object 52 rect from 0.047637, 40 to 5.172342, 50 fc rgb "#00FF00"
set object 53 rect from 0.049821, 40 to 5.229414, 50 fc rgb "#0000FF"
set object 54 rect from 0.050141, 40 to 5.548746, 50 fc rgb "#FFFF00"
set object 55 rect from 0.053212, 40 to 6.091035, 50 fc rgb "#FF00FF"
set object 56 rect from 0.058401, 40 to 7.097846, 50 fc rgb "#808080"
set object 57 rect from 0.068029, 40 to 7.177496, 50 fc rgb "#800080"
set object 58 rect from 0.069055, 40 to 7.243243, 50 fc rgb "#008080"
set object 59 rect from 0.069424, 40 to 144.581173, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.381139, 50 to 146.472494, 60 fc rgb "#FF0000"
set object 61 rect from 0.015084, 50 to 1.834565, 60 fc rgb "#00FF00"
set object 62 rect from 0.017970, 50 to 1.944946, 60 fc rgb "#0000FF"
set object 63 rect from 0.018729, 50 to 2.363267, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022775, 50 to 2.887367, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027747, 50 to 3.912887, 60 fc rgb "#808080"
set object 66 rect from 0.037657, 50 to 4.035812, 60 fc rgb "#800080"
set object 67 rect from 0.039205, 50 to 4.160724, 60 fc rgb "#008080"
set object 68 rect from 0.039943, 50 to 144.269577, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
