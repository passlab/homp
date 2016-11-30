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

set object 15 rect from 0.131970, 0 to 155.948507, 10 fc rgb "#FF0000"
set object 16 rect from 0.101579, 0 to 119.105455, 10 fc rgb "#00FF00"
set object 17 rect from 0.103225, 0 to 119.738465, 10 fc rgb "#0000FF"
set object 18 rect from 0.103569, 0 to 120.437438, 10 fc rgb "#FFFF00"
set object 19 rect from 0.104174, 0 to 121.415306, 10 fc rgb "#FF00FF"
set object 20 rect from 0.105033, 0 to 122.552872, 10 fc rgb "#808080"
set object 21 rect from 0.106005, 0 to 123.053959, 10 fc rgb "#800080"
set object 22 rect from 0.106673, 0 to 123.578188, 10 fc rgb "#008080"
set object 23 rect from 0.106905, 0 to 152.174743, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.130650, 10 to 154.824830, 20 fc rgb "#FF0000"
set object 25 rect from 0.089265, 10 to 104.923477, 20 fc rgb "#00FF00"
set object 26 rect from 0.090977, 10 to 105.650226, 20 fc rgb "#0000FF"
set object 27 rect from 0.091395, 10 to 106.389701, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092063, 10 to 107.548099, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093074, 10 to 108.742369, 20 fc rgb "#808080"
set object 30 rect from 0.094081, 10 to 109.355707, 20 fc rgb "#800080"
set object 31 rect from 0.094847, 10 to 109.963260, 20 fc rgb "#008080"
set object 32 rect from 0.095125, 10 to 150.613628, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129253, 20 to 153.051931, 30 fc rgb "#FF0000"
set object 34 rect from 0.075551, 20 to 88.752206, 30 fc rgb "#00FF00"
set object 35 rect from 0.077000, 20 to 89.424562, 30 fc rgb "#0000FF"
set object 36 rect from 0.077373, 20 to 90.120064, 30 fc rgb "#FFFF00"
set object 37 rect from 0.077976, 20 to 91.500650, 30 fc rgb "#FF00FF"
set object 38 rect from 0.079166, 20 to 92.388253, 30 fc rgb "#808080"
set object 39 rect from 0.079958, 20 to 92.957618, 30 fc rgb "#800080"
set object 40 rect from 0.080683, 20 to 93.643860, 30 fc rgb "#008080"
set object 41 rect from 0.081039, 20 to 148.969189, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127886, 30 to 151.846086, 40 fc rgb "#FF0000"
set object 43 rect from 0.061575, 30 to 72.991756, 40 fc rgb "#00FF00"
set object 44 rect from 0.063395, 30 to 73.909446, 40 fc rgb "#0000FF"
set object 45 rect from 0.063965, 30 to 74.756547, 40 fc rgb "#FFFF00"
set object 46 rect from 0.064715, 30 to 76.075799, 40 fc rgb "#FF00FF"
set object 47 rect from 0.065850, 30 to 76.996963, 40 fc rgb "#808080"
set object 48 rect from 0.066666, 30 to 77.629973, 40 fc rgb "#800080"
set object 49 rect from 0.067419, 30 to 78.306959, 40 fc rgb "#008080"
set object 50 rect from 0.067775, 30 to 147.351367, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126261, 40 to 152.766095, 50 fc rgb "#FF0000"
set object 52 rect from 0.041845, 40 to 52.141761, 50 fc rgb "#00FF00"
set object 53 rect from 0.045454, 40 to 53.447127, 50 fc rgb "#0000FF"
set object 54 rect from 0.046294, 40 to 55.541735, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048138, 40 to 57.568060, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049868, 40 to 58.896572, 50 fc rgb "#808080"
set object 57 rect from 0.051030, 40 to 59.712426, 50 fc rgb "#800080"
set object 58 rect from 0.052093, 40 to 60.963404, 50 fc rgb "#008080"
set object 59 rect from 0.052789, 40 to 145.050777, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.133196, 50 to 157.445978, 60 fc rgb "#FF0000"
set object 61 rect from 0.114567, 50 to 133.952845, 60 fc rgb "#00FF00"
set object 62 rect from 0.116065, 50 to 134.545352, 60 fc rgb "#0000FF"
set object 63 rect from 0.116364, 50 to 135.439896, 60 fc rgb "#FFFF00"
set object 64 rect from 0.117137, 50 to 136.552006, 60 fc rgb "#FF00FF"
set object 65 rect from 0.118118, 50 to 137.437294, 60 fc rgb "#808080"
set object 66 rect from 0.118889, 50 to 138.024016, 60 fc rgb "#800080"
set object 67 rect from 0.119583, 50 to 138.662811, 60 fc rgb "#008080"
set object 68 rect from 0.119924, 50 to 153.654853, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
