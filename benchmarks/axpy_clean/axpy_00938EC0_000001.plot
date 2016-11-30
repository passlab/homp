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

set object 15 rect from 0.331217, 0 to 155.052037, 10 fc rgb "#FF0000"
set object 16 rect from 0.164886, 0 to 73.791183, 10 fc rgb "#00FF00"
set object 17 rect from 0.166836, 0 to 74.116530, 10 fc rgb "#0000FF"
set object 18 rect from 0.167319, 0 to 74.397552, 10 fc rgb "#FFFF00"
set object 19 rect from 0.167964, 0 to 74.849668, 10 fc rgb "#FF00FF"
set object 20 rect from 0.168983, 0 to 82.116328, 10 fc rgb "#808080"
set object 21 rect from 0.185393, 0 to 82.354358, 10 fc rgb "#800080"
set object 22 rect from 0.186169, 0 to 82.585294, 10 fc rgb "#008080"
set object 23 rect from 0.186447, 0 to 146.401538, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.328976, 10 to 154.294071, 20 fc rgb "#FF0000"
set object 25 rect from 0.134329, 10 to 60.326532, 20 fc rgb "#00FF00"
set object 26 rect from 0.136456, 10 to 60.669165, 20 fc rgb "#0000FF"
set object 27 rect from 0.136989, 10 to 61.008250, 20 fc rgb "#FFFF00"
set object 28 rect from 0.137791, 10 to 61.526859, 20 fc rgb "#FF00FF"
set object 29 rect from 0.138929, 10 to 68.887930, 20 fc rgb "#808080"
set object 30 rect from 0.155565, 10 to 69.150781, 20 fc rgb "#800080"
set object 31 rect from 0.156384, 10 to 69.419392, 20 fc rgb "#008080"
set object 32 rect from 0.156766, 10 to 145.507946, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.327147, 20 to 160.851979, 30 fc rgb "#FF0000"
set object 34 rect from 0.084208, 20 to 38.531855, 30 fc rgb "#00FF00"
set object 35 rect from 0.087359, 20 to 39.096560, 30 fc rgb "#0000FF"
set object 36 rect from 0.088335, 20 to 41.082764, 30 fc rgb "#FFFF00"
set object 37 rect from 0.092825, 20 to 46.828639, 30 fc rgb "#FF00FF"
set object 38 rect from 0.105769, 20 to 54.252655, 30 fc rgb "#808080"
set object 39 rect from 0.122610, 20 to 54.780128, 30 fc rgb "#800080"
set object 40 rect from 0.124122, 20 to 55.264153, 30 fc rgb "#008080"
set object 41 rect from 0.124805, 20 to 144.477380, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.332669, 30 to 161.843978, 40 fc rgb "#FF0000"
set object 43 rect from 0.194117, 30 to 86.699108, 40 fc rgb "#00FF00"
set object 44 rect from 0.195964, 30 to 87.015592, 40 fc rgb "#0000FF"
set object 45 rect from 0.196418, 30 to 88.355540, 40 fc rgb "#FFFF00"
set object 46 rect from 0.199457, 30 to 93.546459, 40 fc rgb "#FF00FF"
set object 47 rect from 0.211185, 30 to 100.955846, 40 fc rgb "#808080"
set object 48 rect from 0.227902, 30 to 101.426582, 40 fc rgb "#800080"
set object 49 rect from 0.229207, 30 to 101.764340, 40 fc rgb "#008080"
set object 50 rect from 0.229716, 30 to 147.222000, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.335418, 40 to 162.723832, 50 fc rgb "#FF0000"
set object 52 rect from 0.285538, 40 to 127.128523, 50 fc rgb "#00FF00"
set object 53 rect from 0.287143, 40 to 127.411757, 50 fc rgb "#0000FF"
set object 54 rect from 0.287556, 40 to 128.656409, 50 fc rgb "#FFFF00"
set object 55 rect from 0.290367, 40 to 133.726319, 50 fc rgb "#FF00FF"
set object 56 rect from 0.301805, 40 to 141.049279, 50 fc rgb "#808080"
set object 57 rect from 0.318339, 40 to 141.479229, 50 fc rgb "#800080"
set object 58 rect from 0.319593, 40 to 141.786849, 50 fc rgb "#008080"
set object 59 rect from 0.319997, 40 to 148.483045, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.334069, 50 to 164.794692, 60 fc rgb "#FF0000"
set object 61 rect from 0.237263, 50 to 105.712383, 60 fc rgb "#00FF00"
set object 62 rect from 0.238845, 50 to 106.027532, 60 fc rgb "#0000FF"
set object 63 rect from 0.239311, 50 to 107.363491, 60 fc rgb "#FFFF00"
set object 64 rect from 0.242329, 50 to 115.034844, 60 fc rgb "#FF00FF"
set object 65 rect from 0.259638, 50 to 122.302396, 60 fc rgb "#808080"
set object 66 rect from 0.276049, 50 to 122.742102, 60 fc rgb "#800080"
set object 67 rect from 0.277292, 50 to 123.029332, 60 fc rgb "#008080"
set object 68 rect from 0.277682, 50 to 147.884213, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
