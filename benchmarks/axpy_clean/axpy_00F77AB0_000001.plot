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

set object 15 rect from 0.134679, 0 to 159.275625, 10 fc rgb "#FF0000"
set object 16 rect from 0.113514, 0 to 133.816965, 10 fc rgb "#00FF00"
set object 17 rect from 0.115690, 0 to 134.538727, 10 fc rgb "#0000FF"
set object 18 rect from 0.116061, 0 to 135.400817, 10 fc rgb "#FFFF00"
set object 19 rect from 0.116812, 0 to 136.586718, 10 fc rgb "#FF00FF"
set object 20 rect from 0.117827, 0 to 137.067040, 10 fc rgb "#808080"
set object 21 rect from 0.118253, 0 to 137.651864, 10 fc rgb "#800080"
set object 22 rect from 0.118998, 0 to 138.258751, 10 fc rgb "#008080"
set object 23 rect from 0.119294, 0 to 155.641568, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.133108, 10 to 158.022569, 20 fc rgb "#FF0000"
set object 25 rect from 0.098090, 10 to 115.934046, 20 fc rgb "#00FF00"
set object 26 rect from 0.100283, 10 to 116.725384, 20 fc rgb "#0000FF"
set object 27 rect from 0.100717, 10 to 117.725588, 20 fc rgb "#FFFF00"
set object 28 rect from 0.101610, 10 to 119.121461, 20 fc rgb "#FF00FF"
set object 29 rect from 0.102810, 10 to 119.754975, 20 fc rgb "#808080"
set object 30 rect from 0.103352, 10 to 120.453499, 20 fc rgb "#800080"
set object 31 rect from 0.104194, 10 to 121.154375, 20 fc rgb "#008080"
set object 32 rect from 0.104532, 10 to 153.727058, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129884, 20 to 153.984682, 30 fc rgb "#FF0000"
set object 34 rect from 0.068983, 20 to 82.074556, 30 fc rgb "#00FF00"
set object 35 rect from 0.071110, 20 to 82.785806, 30 fc rgb "#0000FF"
set object 36 rect from 0.071477, 20 to 83.825501, 30 fc rgb "#FFFF00"
set object 37 rect from 0.072359, 20 to 85.103041, 30 fc rgb "#FF00FF"
set object 38 rect from 0.073456, 20 to 85.571813, 30 fc rgb "#808080"
set object 39 rect from 0.073866, 20 to 86.207608, 30 fc rgb "#800080"
set object 40 rect from 0.074675, 20 to 86.915470, 30 fc rgb "#008080"
set object 41 rect from 0.075027, 20 to 150.075503, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.128250, 30 to 152.468054, 40 fc rgb "#FF0000"
set object 43 rect from 0.052962, 30 to 63.842375, 40 fc rgb "#00FF00"
set object 44 rect from 0.055391, 30 to 64.711450, 40 fc rgb "#0000FF"
set object 45 rect from 0.055882, 30 to 65.775420, 40 fc rgb "#FFFF00"
set object 46 rect from 0.056817, 30 to 67.189898, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058030, 30 to 67.718978, 40 fc rgb "#808080"
set object 48 rect from 0.058493, 30 to 68.375730, 40 fc rgb "#800080"
set object 49 rect from 0.059318, 30 to 69.110218, 40 fc rgb "#008080"
set object 50 rect from 0.059694, 30 to 148.123784, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126394, 40 to 153.725951, 50 fc rgb "#FF0000"
set object 52 rect from 0.029727, 40 to 37.858023, 50 fc rgb "#00FF00"
set object 53 rect from 0.033025, 40 to 39.201680, 50 fc rgb "#0000FF"
set object 54 rect from 0.033917, 40 to 42.188322, 50 fc rgb "#FFFF00"
set object 55 rect from 0.036528, 40 to 44.279262, 50 fc rgb "#FF00FF"
set object 56 rect from 0.038293, 40 to 45.079868, 50 fc rgb "#808080"
set object 57 rect from 0.038995, 40 to 45.919965, 50 fc rgb "#800080"
set object 58 rect from 0.040155, 40 to 47.111606, 50 fc rgb "#008080"
set object 59 rect from 0.040737, 40 to 145.409844, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.131395, 50 to 155.604429, 60 fc rgb "#FF0000"
set object 61 rect from 0.084382, 50 to 99.988806, 60 fc rgb "#00FF00"
set object 62 rect from 0.086516, 50 to 100.664161, 60 fc rgb "#0000FF"
set object 63 rect from 0.086869, 50 to 101.675984, 60 fc rgb "#FFFF00"
set object 64 rect from 0.087747, 50 to 102.886161, 60 fc rgb "#FF00FF"
set object 65 rect from 0.088785, 50 to 103.344490, 60 fc rgb "#808080"
set object 66 rect from 0.089178, 50 to 103.932772, 60 fc rgb "#800080"
set object 67 rect from 0.089942, 50 to 104.624311, 60 fc rgb "#008080"
set object 68 rect from 0.090284, 50 to 151.911101, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
