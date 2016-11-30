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

set object 15 rect from 0.158449, 0 to 157.750083, 10 fc rgb "#FF0000"
set object 16 rect from 0.136126, 0 to 134.254969, 10 fc rgb "#00FF00"
set object 17 rect from 0.138404, 0 to 134.997031, 10 fc rgb "#0000FF"
set object 18 rect from 0.138923, 0 to 135.777033, 10 fc rgb "#FFFF00"
set object 19 rect from 0.139725, 0 to 136.810853, 10 fc rgb "#FF00FF"
set object 20 rect from 0.140796, 0 to 138.115070, 10 fc rgb "#808080"
set object 21 rect from 0.142126, 0 to 138.603285, 10 fc rgb "#800080"
set object 22 rect from 0.142883, 0 to 139.127499, 10 fc rgb "#008080"
set object 23 rect from 0.143172, 0 to 153.517473, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.156770, 10 to 156.469141, 20 fc rgb "#FF0000"
set object 25 rect from 0.119688, 10 to 118.299164, 20 fc rgb "#00FF00"
set object 26 rect from 0.122086, 10 to 119.167656, 20 fc rgb "#0000FF"
set object 27 rect from 0.122644, 10 to 119.899023, 20 fc rgb "#FFFF00"
set object 28 rect from 0.123433, 10 to 121.175008, 20 fc rgb "#FF00FF"
set object 29 rect from 0.124735, 10 to 122.537571, 20 fc rgb "#808080"
set object 30 rect from 0.126133, 10 to 123.138623, 20 fc rgb "#800080"
set object 31 rect from 0.126989, 10 to 123.707560, 20 fc rgb "#008080"
set object 32 rect from 0.127317, 10 to 151.840777, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.154983, 20 to 154.644652, 30 fc rgb "#FF0000"
set object 34 rect from 0.104151, 20 to 103.040698, 30 fc rgb "#00FF00"
set object 35 rect from 0.106295, 20 to 103.781774, 30 fc rgb "#0000FF"
set object 36 rect from 0.106822, 20 to 104.629832, 30 fc rgb "#FFFF00"
set object 37 rect from 0.107700, 20 to 105.869848, 30 fc rgb "#FF00FF"
set object 38 rect from 0.108969, 20 to 107.044706, 30 fc rgb "#808080"
set object 39 rect from 0.110184, 20 to 107.614629, 30 fc rgb "#800080"
set object 40 rect from 0.111042, 20 to 108.238028, 30 fc rgb "#008080"
set object 41 rect from 0.111428, 20 to 150.244802, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.153452, 30 to 153.062358, 40 fc rgb "#FF0000"
set object 43 rect from 0.087871, 30 to 87.210338, 40 fc rgb "#00FF00"
set object 44 rect from 0.090051, 30 to 87.903763, 40 fc rgb "#0000FF"
set object 45 rect from 0.090497, 30 to 88.801443, 40 fc rgb "#FFFF00"
set object 46 rect from 0.091431, 30 to 89.963664, 40 fc rgb "#FF00FF"
set object 47 rect from 0.092617, 30 to 91.114205, 40 fc rgb "#808080"
set object 48 rect from 0.093802, 30 to 91.749285, 40 fc rgb "#800080"
set object 49 rect from 0.094716, 30 to 92.345438, 40 fc rgb "#008080"
set object 50 rect from 0.095086, 30 to 148.731520, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.151891, 40 to 151.826198, 50 fc rgb "#FF0000"
set object 52 rect from 0.071402, 40 to 71.441251, 50 fc rgb "#00FF00"
set object 53 rect from 0.073911, 40 to 72.336032, 50 fc rgb "#0000FF"
set object 54 rect from 0.074491, 40 to 73.227857, 50 fc rgb "#FFFF00"
set object 55 rect from 0.075415, 40 to 74.534972, 50 fc rgb "#FF00FF"
set object 56 rect from 0.076786, 40 to 75.763307, 50 fc rgb "#808080"
set object 57 rect from 0.078057, 40 to 76.396445, 50 fc rgb "#800080"
set object 58 rect from 0.078937, 40 to 77.029583, 50 fc rgb "#008080"
set object 59 rect from 0.079336, 40 to 147.095663, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.150002, 50 to 153.167282, 60 fc rgb "#FF0000"
set object 61 rect from 0.047573, 50 to 49.578140, 60 fc rgb "#00FF00"
set object 62 rect from 0.051505, 50 to 51.071973, 60 fc rgb "#0000FF"
set object 63 rect from 0.052642, 50 to 53.392503, 60 fc rgb "#FFFF00"
set object 64 rect from 0.055060, 50 to 55.479620, 60 fc rgb "#FF00FF"
set object 65 rect from 0.057176, 50 to 57.024060, 60 fc rgb "#808080"
set object 66 rect from 0.058779, 50 to 57.786527, 60 fc rgb "#800080"
set object 67 rect from 0.059999, 50 to 59.044049, 60 fc rgb "#008080"
set object 68 rect from 0.060839, 50 to 145.068862, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
