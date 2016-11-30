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

set object 15 rect from 0.125286, 0 to 161.738106, 10 fc rgb "#FF0000"
set object 16 rect from 0.103703, 0 to 132.692322, 10 fc rgb "#00FF00"
set object 17 rect from 0.105953, 0 to 133.457658, 10 fc rgb "#0000FF"
set object 18 rect from 0.106309, 0 to 134.354938, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107026, 0 to 135.635465, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108044, 0 to 137.212615, 10 fc rgb "#808080"
set object 21 rect from 0.109330, 0 to 137.872372, 10 fc rgb "#800080"
set object 22 rect from 0.110085, 0 to 138.518310, 10 fc rgb "#008080"
set object 23 rect from 0.110341, 0 to 156.723932, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.123641, 10 to 160.093094, 20 fc rgb "#FF0000"
set object 25 rect from 0.088879, 10 to 114.113452, 20 fc rgb "#00FF00"
set object 26 rect from 0.091242, 10 to 115.033353, 20 fc rgb "#0000FF"
set object 27 rect from 0.091647, 10 to 116.077670, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092514, 10 to 117.574371, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093681, 10 to 119.159049, 20 fc rgb "#808080"
set object 30 rect from 0.094948, 10 to 119.870341, 20 fc rgb "#800080"
set object 31 rect from 0.095752, 10 to 120.606763, 20 fc rgb "#008080"
set object 32 rect from 0.096089, 10 to 154.619007, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121919, 20 to 157.669001, 30 fc rgb "#FF0000"
set object 34 rect from 0.075084, 20 to 96.581408, 30 fc rgb "#00FF00"
set object 35 rect from 0.077226, 20 to 97.327905, 30 fc rgb "#0000FF"
set object 36 rect from 0.077560, 20 to 98.409899, 30 fc rgb "#FFFF00"
set object 37 rect from 0.078421, 20 to 99.857613, 30 fc rgb "#FF00FF"
set object 38 rect from 0.079572, 20 to 101.290234, 30 fc rgb "#808080"
set object 39 rect from 0.080717, 20 to 101.933663, 30 fc rgb "#800080"
set object 40 rect from 0.081478, 20 to 102.715327, 30 fc rgb "#008080"
set object 41 rect from 0.081850, 20 to 152.603255, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.120364, 30 to 156.032752, 40 fc rgb "#FF0000"
set object 43 rect from 0.059586, 30 to 77.096494, 40 fc rgb "#00FF00"
set object 44 rect from 0.061699, 30 to 77.928420, 40 fc rgb "#0000FF"
set object 45 rect from 0.062141, 30 to 79.104680, 40 fc rgb "#FFFF00"
set object 46 rect from 0.063060, 30 to 80.536028, 40 fc rgb "#FF00FF"
set object 47 rect from 0.064201, 30 to 81.998836, 40 fc rgb "#808080"
set object 48 rect from 0.065363, 30 to 82.761625, 40 fc rgb "#800080"
set object 49 rect from 0.066235, 30 to 83.548308, 40 fc rgb "#008080"
set object 50 rect from 0.066615, 30 to 150.443013, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.118701, 40 to 154.045951, 50 fc rgb "#FF0000"
set object 52 rect from 0.044233, 40 to 58.244934, 50 fc rgb "#00FF00"
set object 53 rect from 0.046787, 40 to 59.137195, 50 fc rgb "#0000FF"
set object 54 rect from 0.047170, 40 to 60.310909, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048105, 40 to 61.818959, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049319, 40 to 63.338282, 50 fc rgb "#808080"
set object 57 rect from 0.050525, 40 to 64.107363, 50 fc rgb "#800080"
set object 58 rect from 0.051436, 40 to 65.027264, 50 fc rgb "#008080"
set object 59 rect from 0.051873, 40 to 148.408460, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116884, 50 to 154.650429, 60 fc rgb "#FF0000"
set object 61 rect from 0.023242, 50 to 33.396455, 60 fc rgb "#00FF00"
set object 62 rect from 0.027109, 50 to 35.000008, 60 fc rgb "#0000FF"
set object 63 rect from 0.028020, 50 to 37.344964, 60 fc rgb "#FFFF00"
set object 64 rect from 0.029873, 50 to 39.640933, 60 fc rgb "#FF00FF"
set object 65 rect from 0.031657, 50 to 41.388977, 60 fc rgb "#808080"
set object 66 rect from 0.033072, 50 to 42.386816, 60 fc rgb "#800080"
set object 67 rect from 0.034288, 50 to 43.971493, 60 fc rgb "#008080"
set object 68 rect from 0.035125, 50 to 145.632429, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
