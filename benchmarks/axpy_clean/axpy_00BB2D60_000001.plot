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

set object 15 rect from 0.247043, 0 to 153.635433, 10 fc rgb "#FF0000"
set object 16 rect from 0.138155, 0 to 83.420834, 10 fc rgb "#00FF00"
set object 17 rect from 0.140076, 0 to 83.804126, 10 fc rgb "#0000FF"
set object 18 rect from 0.140476, 0 to 84.159956, 10 fc rgb "#FFFF00"
set object 19 rect from 0.141089, 0 to 84.702654, 10 fc rgb "#FF00FF"
set object 20 rect from 0.141981, 0 to 89.661561, 10 fc rgb "#808080"
set object 21 rect from 0.150297, 0 to 89.979180, 10 fc rgb "#800080"
set object 22 rect from 0.151062, 0 to 90.270528, 10 fc rgb "#008080"
set object 23 rect from 0.151309, 0 to 147.113500, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.245277, 10 to 152.822267, 20 fc rgb "#FF0000"
set object 25 rect from 0.115782, 10 to 70.211005, 20 fc rgb "#00FF00"
set object 26 rect from 0.117953, 10 to 70.677283, 20 fc rgb "#0000FF"
set object 27 rect from 0.118490, 10 to 71.101179, 20 fc rgb "#FFFF00"
set object 28 rect from 0.119225, 10 to 71.776416, 20 fc rgb "#FF00FF"
set object 29 rect from 0.120370, 10 to 76.725172, 20 fc rgb "#808080"
set object 30 rect from 0.128632, 10 to 77.045175, 20 fc rgb "#800080"
set object 31 rect from 0.129454, 10 to 77.412952, 20 fc rgb "#008080"
set object 32 rect from 0.129783, 10 to 146.026315, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.250092, 20 to 159.107173, 30 fc rgb "#FF0000"
set object 34 rect from 0.188096, 20 to 113.040554, 30 fc rgb "#00FF00"
set object 35 rect from 0.189667, 20 to 113.404141, 30 fc rgb "#0000FF"
set object 36 rect from 0.190055, 20 to 115.138512, 30 fc rgb "#FFFF00"
set object 37 rect from 0.192984, 20 to 118.114687, 30 fc rgb "#FF00FF"
set object 38 rect from 0.197964, 20 to 122.906430, 30 fc rgb "#808080"
set object 39 rect from 0.205976, 20 to 123.252109, 30 fc rgb "#800080"
set object 40 rect from 0.206801, 20 to 123.679581, 30 fc rgb "#008080"
set object 41 rect from 0.207272, 20 to 148.972643, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.251447, 30 to 161.064853, 40 fc rgb "#FF0000"
set object 43 rect from 0.215476, 30 to 129.382983, 40 fc rgb "#00FF00"
set object 44 rect from 0.217042, 30 to 129.733439, 40 fc rgb "#0000FF"
set object 45 rect from 0.217405, 30 to 131.508395, 40 fc rgb "#FFFF00"
set object 46 rect from 0.220382, 30 to 134.525182, 40 fc rgb "#FF00FF"
set object 47 rect from 0.225442, 30 to 140.345003, 40 fc rgb "#808080"
set object 48 rect from 0.235189, 30 to 140.727699, 40 fc rgb "#800080"
set object 49 rect from 0.236117, 30 to 141.110983, 40 fc rgb "#008080"
set object 50 rect from 0.236469, 30 to 149.849677, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.248632, 40 to 158.556708, 50 fc rgb "#FF0000"
set object 52 rect from 0.160477, 40 to 96.726780, 50 fc rgb "#00FF00"
set object 53 rect from 0.162357, 40 to 97.106487, 50 fc rgb "#0000FF"
set object 54 rect from 0.162766, 40 to 98.840262, 50 fc rgb "#FFFF00"
set object 55 rect from 0.165665, 40 to 102.051667, 50 fc rgb "#FF00FF"
set object 56 rect from 0.171073, 40 to 106.876843, 50 fc rgb "#808080"
set object 57 rect from 0.179150, 40 to 107.284013, 50 fc rgb "#800080"
set object 58 rect from 0.180043, 40 to 107.697758, 50 fc rgb "#008080"
set object 59 rect from 0.180519, 40 to 148.092620, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.243192, 50 to 157.522066, 60 fc rgb "#FF0000"
set object 61 rect from 0.081032, 50 to 49.997449, 60 fc rgb "#00FF00"
set object 62 rect from 0.084180, 50 to 50.770004, 60 fc rgb "#0000FF"
set object 63 rect from 0.085161, 50 to 53.524098, 60 fc rgb "#FFFF00"
set object 64 rect from 0.089812, 50 to 57.426264, 60 fc rgb "#FF00FF"
set object 65 rect from 0.096304, 50 to 62.366666, 60 fc rgb "#808080"
set object 66 rect from 0.104596, 50 to 62.834137, 60 fc rgb "#800080"
set object 67 rect from 0.105794, 50 to 63.511767, 60 fc rgb "#008080"
set object 68 rect from 0.106494, 50 to 144.562404, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
