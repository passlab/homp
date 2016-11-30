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

set object 15 rect from 0.147586, 0 to 160.678294, 10 fc rgb "#FF0000"
set object 16 rect from 0.123017, 0 to 132.739612, 10 fc rgb "#00FF00"
set object 17 rect from 0.125448, 0 to 133.480540, 10 fc rgb "#0000FF"
set object 18 rect from 0.125867, 0 to 134.366895, 10 fc rgb "#FFFF00"
set object 19 rect from 0.126702, 0 to 135.536669, 10 fc rgb "#FF00FF"
set object 20 rect from 0.127805, 0 to 136.984549, 10 fc rgb "#808080"
set object 21 rect from 0.129193, 0 to 137.585364, 10 fc rgb "#800080"
set object 22 rect from 0.130020, 0 to 138.198897, 10 fc rgb "#008080"
set object 23 rect from 0.130333, 0 to 155.942949, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.145820, 10 to 159.168884, 20 fc rgb "#FF0000"
set object 25 rect from 0.106257, 10 to 115.046525, 20 fc rgb "#00FF00"
set object 26 rect from 0.108759, 10 to 115.862808, 20 fc rgb "#0000FF"
set object 27 rect from 0.109268, 10 to 116.718382, 20 fc rgb "#FFFF00"
set object 28 rect from 0.110132, 10 to 118.135480, 20 fc rgb "#FF00FF"
set object 29 rect from 0.111432, 10 to 119.618411, 20 fc rgb "#808080"
set object 30 rect from 0.112832, 10 to 120.312614, 20 fc rgb "#800080"
set object 31 rect from 0.113783, 10 to 121.053541, 20 fc rgb "#008080"
set object 32 rect from 0.114169, 10 to 154.050310, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.143928, 20 to 157.262421, 30 fc rgb "#FF0000"
set object 34 rect from 0.089947, 20 to 97.512658, 30 fc rgb "#00FF00"
set object 35 rect from 0.092230, 20 to 98.300279, 30 fc rgb "#0000FF"
set object 36 rect from 0.092728, 20 to 99.260945, 30 fc rgb "#FFFF00"
set object 37 rect from 0.093634, 20 to 100.727932, 30 fc rgb "#FF00FF"
set object 38 rect from 0.095025, 20 to 102.094098, 30 fc rgb "#808080"
set object 39 rect from 0.096316, 20 to 102.804213, 30 fc rgb "#800080"
set object 40 rect from 0.097278, 20 to 103.509076, 30 fc rgb "#008080"
set object 41 rect from 0.097639, 20 to 152.131098, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.142073, 30 to 155.373990, 40 fc rgb "#FF0000"
set object 43 rect from 0.072123, 30 to 78.745303, 40 fc rgb "#00FF00"
set object 44 rect from 0.074561, 30 to 79.531880, 40 fc rgb "#0000FF"
set object 45 rect from 0.075077, 30 to 80.599758, 40 fc rgb "#FFFF00"
set object 46 rect from 0.076055, 30 to 82.064625, 40 fc rgb "#FF00FF"
set object 47 rect from 0.077444, 30 to 83.438193, 40 fc rgb "#808080"
set object 48 rect from 0.078739, 30 to 84.157894, 40 fc rgb "#800080"
set object 49 rect from 0.079714, 30 to 84.868041, 40 fc rgb "#008080"
set object 50 rect from 0.080098, 30 to 149.838274, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.139987, 40 to 153.281765, 50 fc rgb "#FF0000"
set object 52 rect from 0.054979, 40 to 60.801696, 50 fc rgb "#00FF00"
set object 53 rect from 0.057712, 40 to 61.763406, 50 fc rgb "#0000FF"
set object 54 rect from 0.058323, 40 to 62.782471, 50 fc rgb "#FFFF00"
set object 55 rect from 0.059269, 40 to 64.265370, 50 fc rgb "#FF00FF"
set object 56 rect from 0.060712, 40 to 65.678229, 50 fc rgb "#808080"
set object 57 rect from 0.062045, 40 to 66.452058, 50 fc rgb "#800080"
set object 58 rect from 0.063051, 40 to 67.213169, 50 fc rgb "#008080"
set object 59 rect from 0.063455, 40 to 147.818241, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.137946, 50 to 154.696744, 60 fc rgb "#FF0000"
set object 61 rect from 0.030283, 50 to 35.504016, 60 fc rgb "#00FF00"
set object 62 rect from 0.033933, 50 to 37.061226, 60 fc rgb "#0000FF"
set object 63 rect from 0.035075, 50 to 40.160810, 60 fc rgb "#FFFF00"
set object 64 rect from 0.038032, 50 to 42.224374, 60 fc rgb "#FF00FF"
set object 65 rect from 0.039922, 50 to 43.935521, 60 fc rgb "#808080"
set object 66 rect from 0.041559, 50 to 44.851582, 60 fc rgb "#800080"
set object 67 rect from 0.043051, 50 to 46.373772, 60 fc rgb "#008080"
set object 68 rect from 0.043837, 50 to 145.283376, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
