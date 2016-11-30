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

set object 15 rect from 0.138663, 0 to 159.531789, 10 fc rgb "#FF0000"
set object 16 rect from 0.116289, 0 to 132.572177, 10 fc rgb "#00FF00"
set object 17 rect from 0.118522, 0 to 133.292446, 10 fc rgb "#0000FF"
set object 18 rect from 0.118923, 0 to 134.161930, 10 fc rgb "#FFFF00"
set object 19 rect from 0.119698, 0 to 135.286089, 10 fc rgb "#FF00FF"
set object 20 rect from 0.120699, 0 to 136.768138, 10 fc rgb "#808080"
set object 21 rect from 0.122031, 0 to 137.336949, 10 fc rgb "#800080"
set object 22 rect from 0.122795, 0 to 137.954002, 10 fc rgb "#008080"
set object 23 rect from 0.123078, 0 to 154.901646, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.137046, 10 to 158.135005, 20 fc rgb "#FF0000"
set object 25 rect from 0.099996, 10 to 114.365745, 20 fc rgb "#00FF00"
set object 26 rect from 0.102275, 10 to 115.152207, 20 fc rgb "#0000FF"
set object 27 rect from 0.102750, 10 to 116.027300, 20 fc rgb "#FFFF00"
set object 28 rect from 0.103564, 10 to 117.396037, 20 fc rgb "#FF00FF"
set object 29 rect from 0.104771, 10 to 118.925206, 20 fc rgb "#808080"
set object 30 rect from 0.106138, 10 to 119.588258, 20 fc rgb "#800080"
set object 31 rect from 0.106995, 10 to 120.287210, 20 fc rgb "#008080"
set object 32 rect from 0.107335, 10 to 153.065071, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.135312, 20 to 156.084143, 30 fc rgb "#FF0000"
set object 34 rect from 0.084625, 20 to 96.990651, 30 fc rgb "#00FF00"
set object 35 rect from 0.086809, 20 to 97.740090, 30 fc rgb "#0000FF"
set object 36 rect from 0.087232, 20 to 98.671279, 30 fc rgb "#FFFF00"
set object 37 rect from 0.088061, 20 to 100.051234, 30 fc rgb "#FF00FF"
set object 38 rect from 0.089293, 20 to 101.341436, 30 fc rgb "#808080"
set object 39 rect from 0.090444, 20 to 102.050487, 30 fc rgb "#800080"
set object 40 rect from 0.091331, 20 to 102.710172, 30 fc rgb "#008080"
set object 41 rect from 0.091663, 20 to 151.261032, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133806, 30 to 154.930817, 40 fc rgb "#FF0000"
set object 43 rect from 0.067808, 30 to 78.096485, 40 fc rgb "#00FF00"
set object 44 rect from 0.069959, 30 to 79.226253, 40 fc rgb "#0000FF"
set object 45 rect from 0.070732, 30 to 80.229244, 40 fc rgb "#FFFF00"
set object 46 rect from 0.071624, 30 to 81.691100, 40 fc rgb "#FF00FF"
set object 47 rect from 0.072944, 30 to 83.011593, 40 fc rgb "#808080"
set object 48 rect from 0.074106, 30 to 83.695962, 40 fc rgb "#800080"
set object 49 rect from 0.074973, 30 to 84.376964, 40 fc rgb "#008080"
set object 50 rect from 0.075341, 30 to 149.502992, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.132079, 40 to 152.735229, 50 fc rgb "#FF0000"
set object 52 rect from 0.051547, 40 to 60.218773, 50 fc rgb "#00FF00"
set object 53 rect from 0.054033, 40 to 61.036649, 50 fc rgb "#0000FF"
set object 54 rect from 0.054516, 40 to 62.036275, 50 fc rgb "#FFFF00"
set object 55 rect from 0.055410, 40 to 63.539641, 50 fc rgb "#FF00FF"
set object 56 rect from 0.056759, 40 to 64.873598, 50 fc rgb "#808080"
set object 57 rect from 0.057960, 40 to 65.597233, 50 fc rgb "#800080"
set object 58 rect from 0.058881, 40 to 66.373598, 50 fc rgb "#008080"
set object 59 rect from 0.059288, 40 to 147.479058, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.130254, 50 to 154.201571, 60 fc rgb "#FF0000"
set object 61 rect from 0.029138, 50 to 35.924832, 60 fc rgb "#00FF00"
set object 62 rect from 0.032488, 50 to 37.327225, 60 fc rgb "#0000FF"
set object 63 rect from 0.033404, 50 to 40.127525, 60 fc rgb "#FFFF00"
set object 64 rect from 0.035912, 50 to 42.425206, 60 fc rgb "#FF00FF"
set object 65 rect from 0.037935, 50 to 44.074420, 60 fc rgb "#808080"
set object 66 rect from 0.039406, 50 to 44.925953, 60 fc rgb "#800080"
set object 67 rect from 0.040578, 50 to 46.182498, 60 fc rgb "#008080"
set object 68 rect from 0.041287, 50 to 145.151085, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
