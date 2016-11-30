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

set object 15 rect from 0.159791, 0 to 157.023468, 10 fc rgb "#FF0000"
set object 16 rect from 0.069619, 0 to 66.213081, 10 fc rgb "#00FF00"
set object 17 rect from 0.071461, 0 to 66.855514, 10 fc rgb "#0000FF"
set object 18 rect from 0.071916, 0 to 67.418784, 10 fc rgb "#FFFF00"
set object 19 rect from 0.072518, 0 to 68.300482, 10 fc rgb "#FF00FF"
set object 20 rect from 0.073481, 0 to 74.635327, 10 fc rgb "#808080"
set object 21 rect from 0.080299, 0 to 75.124123, 10 fc rgb "#800080"
set object 22 rect from 0.081047, 0 to 75.576598, 10 fc rgb "#008080"
set object 23 rect from 0.081283, 0 to 148.191561, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.158235, 10 to 157.112814, 20 fc rgb "#FF0000"
set object 25 rect from 0.049186, 10 to 47.526086, 20 fc rgb "#00FF00"
set object 26 rect from 0.051376, 10 to 49.222444, 20 fc rgb "#0000FF"
set object 27 rect from 0.052978, 10 to 49.845315, 20 fc rgb "#FFFF00"
set object 28 rect from 0.053681, 10 to 50.901127, 20 fc rgb "#FF00FF"
set object 29 rect from 0.054811, 10 to 57.451014, 20 fc rgb "#808080"
set object 30 rect from 0.061843, 10 to 58.014311, 20 fc rgb "#800080"
set object 31 rect from 0.062688, 10 to 58.562681, 20 fc rgb "#008080"
set object 32 rect from 0.063020, 10 to 146.712131, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.156548, 20 to 162.661828, 30 fc rgb "#FF0000"
set object 34 rect from 0.018113, 20 to 19.478376, 30 fc rgb "#00FF00"
set object 35 rect from 0.021355, 20 to 22.753777, 30 fc rgb "#0000FF"
set object 36 rect from 0.024559, 20 to 26.544084, 30 fc rgb "#FFFF00"
set object 37 rect from 0.028650, 20 to 30.016880, 30 fc rgb "#FF00FF"
set object 38 rect from 0.032363, 20 to 36.435494, 30 fc rgb "#808080"
set object 39 rect from 0.039287, 20 to 37.178455, 30 fc rgb "#800080"
set object 40 rect from 0.040495, 20 to 38.273363, 30 fc rgb "#008080"
set object 41 rect from 0.041226, 20 to 144.853756, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.162687, 30 to 163.713922, 40 fc rgb "#FF0000"
set object 43 rect from 0.112518, 30 to 105.946340, 40 fc rgb "#00FF00"
set object 44 rect from 0.114117, 30 to 106.469598, 40 fc rgb "#0000FF"
set object 45 rect from 0.114464, 30 to 109.096989, 40 fc rgb "#FFFF00"
set object 46 rect from 0.117322, 30 to 112.055849, 40 fc rgb "#FF00FF"
set object 47 rect from 0.120463, 30 to 118.252901, 40 fc rgb "#808080"
set object 48 rect from 0.127144, 30 to 118.853408, 40 fc rgb "#800080"
set object 49 rect from 0.127998, 30 to 119.417621, 40 fc rgb "#008080"
set object 50 rect from 0.128369, 30 to 151.001446, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.164041, 40 to 164.913105, 50 fc rgb "#FF0000"
set object 52 rect from 0.135524, 40 to 127.364085, 50 fc rgb "#00FF00"
set object 53 rect from 0.137123, 40 to 127.853824, 50 fc rgb "#0000FF"
set object 54 rect from 0.137431, 40 to 130.455160, 50 fc rgb "#FFFF00"
set object 55 rect from 0.140225, 40 to 133.367460, 50 fc rgb "#FF00FF"
set object 56 rect from 0.143390, 40 to 139.598031, 50 fc rgb "#808080"
set object 57 rect from 0.150055, 40 to 140.198538, 50 fc rgb "#800080"
set object 58 rect from 0.150928, 40 to 140.727374, 50 fc rgb "#008080"
set object 59 rect from 0.151260, 40 to 152.316085, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.161234, 50 to 162.834999, 60 fc rgb "#FF0000"
set object 61 rect from 0.089165, 50 to 84.448516, 60 fc rgb "#00FF00"
set object 62 rect from 0.091026, 50 to 85.108624, 60 fc rgb "#0000FF"
set object 63 rect from 0.091524, 50 to 87.874780, 60 fc rgb "#FFFF00"
set object 64 rect from 0.094513, 50 to 90.959307, 60 fc rgb "#FF00FF"
set object 65 rect from 0.097839, 50 to 97.229917, 60 fc rgb "#808080"
set object 66 rect from 0.104574, 50 to 97.861140, 60 fc rgb "#800080"
set object 67 rect from 0.105459, 50 to 98.470998, 60 fc rgb "#008080"
set object 68 rect from 0.105875, 50 to 149.605813, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
