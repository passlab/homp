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

set object 15 rect from 0.109530, 0 to 163.199704, 10 fc rgb "#FF0000"
set object 16 rect from 0.089667, 0 to 131.885669, 10 fc rgb "#00FF00"
set object 17 rect from 0.091501, 0 to 132.824884, 10 fc rgb "#0000FF"
set object 18 rect from 0.091910, 0 to 133.876990, 10 fc rgb "#FFFF00"
set object 19 rect from 0.092629, 0 to 135.369025, 10 fc rgb "#FF00FF"
set object 20 rect from 0.093659, 0 to 136.890011, 10 fc rgb "#808080"
set object 21 rect from 0.094725, 0 to 137.657011, 10 fc rgb "#800080"
set object 22 rect from 0.095544, 0 to 138.471777, 10 fc rgb "#008080"
set object 23 rect from 0.095824, 0 to 157.701876, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.107992, 10 to 161.513749, 20 fc rgb "#FF0000"
set object 25 rect from 0.075650, 10 to 111.738059, 20 fc rgb "#00FF00"
set object 26 rect from 0.077590, 10 to 112.755425, 20 fc rgb "#0000FF"
set object 27 rect from 0.078044, 10 to 113.819100, 20 fc rgb "#FFFF00"
set object 28 rect from 0.078803, 10 to 115.610701, 20 fc rgb "#FF00FF"
set object 29 rect from 0.080043, 10 to 117.263382, 20 fc rgb "#808080"
set object 30 rect from 0.081177, 10 to 118.196807, 20 fc rgb "#800080"
set object 31 rect from 0.082066, 10 to 119.121563, 20 fc rgb "#008080"
set object 32 rect from 0.082455, 10 to 155.312591, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.106280, 20 to 158.329957, 30 fc rgb "#FF0000"
set object 34 rect from 0.063185, 20 to 93.447175, 30 fc rgb "#00FF00"
set object 35 rect from 0.064942, 20 to 94.302461, 30 fc rgb "#0000FF"
set object 36 rect from 0.065294, 20 to 95.412446, 30 fc rgb "#FFFF00"
set object 37 rect from 0.066050, 20 to 96.986967, 30 fc rgb "#FF00FF"
set object 38 rect from 0.067148, 20 to 98.267722, 30 fc rgb "#808080"
set object 39 rect from 0.068049, 20 to 99.095513, 30 fc rgb "#800080"
set object 40 rect from 0.068867, 20 to 99.984073, 30 fc rgb "#008080"
set object 41 rect from 0.069227, 20 to 153.120114, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.104812, 30 to 156.520975, 40 fc rgb "#FF0000"
set object 43 rect from 0.048927, 30 to 72.956580, 40 fc rgb "#00FF00"
set object 44 rect from 0.050798, 30 to 73.806076, 40 fc rgb "#0000FF"
set object 45 rect from 0.051128, 30 to 75.060781, 40 fc rgb "#FFFF00"
set object 46 rect from 0.052000, 30 to 76.732267, 40 fc rgb "#FF00FF"
set object 47 rect from 0.053165, 30 to 78.123002, 40 fc rgb "#808080"
set object 48 rect from 0.054105, 30 to 78.946448, 40 fc rgb "#800080"
set object 49 rect from 0.054975, 30 to 79.913169, 40 fc rgb "#008080"
set object 50 rect from 0.055372, 30 to 150.846598, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.103251, 40 to 154.306816, 50 fc rgb "#FF0000"
set object 52 rect from 0.035039, 40 to 53.358901, 50 fc rgb "#00FF00"
set object 53 rect from 0.037259, 40 to 54.286536, 50 fc rgb "#0000FF"
set object 54 rect from 0.037642, 40 to 55.474672, 50 fc rgb "#FFFF00"
set object 55 rect from 0.038482, 40 to 57.177997, 50 fc rgb "#FF00FF"
set object 56 rect from 0.039648, 40 to 58.532558, 50 fc rgb "#808080"
set object 57 rect from 0.040586, 40 to 59.441389, 50 fc rgb "#800080"
set object 58 rect from 0.041539, 40 to 60.588994, 50 fc rgb "#008080"
set object 59 rect from 0.042017, 40 to 148.516637, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.101468, 50 to 154.913160, 60 fc rgb "#FF0000"
set object 61 rect from 0.016581, 50 to 27.674378, 60 fc rgb "#00FF00"
set object 62 rect from 0.019597, 50 to 29.280749, 60 fc rgb "#0000FF"
set object 63 rect from 0.020365, 50 to 32.070911, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022339, 50 to 34.384947, 60 fc rgb "#FF00FF"
set object 65 rect from 0.023891, 50 to 36.027492, 60 fc rgb "#808080"
set object 66 rect from 0.025042, 50 to 37.086822, 60 fc rgb "#800080"
set object 67 rect from 0.026205, 50 to 38.826334, 60 fc rgb "#008080"
set object 68 rect from 0.026972, 50 to 145.541235, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
