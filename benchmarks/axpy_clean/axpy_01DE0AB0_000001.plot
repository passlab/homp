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

set object 15 rect from 0.145459, 0 to 157.919637, 10 fc rgb "#FF0000"
set object 16 rect from 0.124351, 0 to 134.605839, 10 fc rgb "#00FF00"
set object 17 rect from 0.126661, 0 to 135.270913, 10 fc rgb "#0000FF"
set object 18 rect from 0.127030, 0 to 136.066028, 10 fc rgb "#FFFF00"
set object 19 rect from 0.127796, 0 to 137.256542, 10 fc rgb "#FF00FF"
set object 20 rect from 0.128894, 0 to 137.728681, 10 fc rgb "#808080"
set object 21 rect from 0.129372, 0 to 138.304245, 10 fc rgb "#800080"
set object 22 rect from 0.130156, 0 to 138.888320, 10 fc rgb "#008080"
set object 23 rect from 0.130448, 0 to 154.435447, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.143830, 10 to 156.552134, 20 fc rgb "#FF0000"
set object 25 rect from 0.109193, 10 to 118.438393, 20 fc rgb "#00FF00"
set object 26 rect from 0.111557, 10 to 119.300626, 20 fc rgb "#0000FF"
set object 27 rect from 0.112046, 10 to 120.135161, 20 fc rgb "#FFFF00"
set object 28 rect from 0.112872, 10 to 121.448252, 20 fc rgb "#FF00FF"
set object 29 rect from 0.114087, 10 to 121.955586, 20 fc rgb "#808080"
set object 30 rect from 0.114572, 10 to 122.624948, 20 fc rgb "#800080"
set object 31 rect from 0.115423, 10 to 123.266548, 20 fc rgb "#008080"
set object 32 rect from 0.115776, 10 to 152.664066, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.142192, 20 to 154.535599, 30 fc rgb "#FF0000"
set object 34 rect from 0.095481, 20 to 103.742827, 30 fc rgb "#00FF00"
set object 35 rect from 0.097686, 20 to 104.387667, 30 fc rgb "#0000FF"
set object 36 rect from 0.098057, 20 to 105.330898, 30 fc rgb "#FFFF00"
set object 37 rect from 0.098942, 20 to 106.481992, 30 fc rgb "#FF00FF"
set object 38 rect from 0.100021, 20 to 106.928562, 30 fc rgb "#808080"
set object 39 rect from 0.100464, 20 to 107.494532, 30 fc rgb "#800080"
set object 40 rect from 0.101233, 20 to 108.112691, 30 fc rgb "#008080"
set object 41 rect from 0.101558, 20 to 150.867083, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.140489, 30 to 152.756754, 40 fc rgb "#FF0000"
set object 43 rect from 0.080204, 30 to 87.471989, 40 fc rgb "#00FF00"
set object 44 rect from 0.082516, 30 to 88.181850, 40 fc rgb "#0000FF"
set object 45 rect from 0.082866, 30 to 89.194359, 40 fc rgb "#FFFF00"
set object 46 rect from 0.083799, 30 to 90.336940, 40 fc rgb "#FF00FF"
set object 47 rect from 0.084872, 30 to 90.787766, 40 fc rgb "#808080"
set object 48 rect from 0.085300, 30 to 91.352657, 40 fc rgb "#800080"
set object 49 rect from 0.086080, 30 to 91.976151, 40 fc rgb "#008080"
set object 50 rect from 0.086415, 30 to 149.142586, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.138870, 40 to 151.231607, 50 fc rgb "#FF0000"
set object 52 rect from 0.065099, 40 to 71.864098, 50 fc rgb "#00FF00"
set object 53 rect from 0.067779, 40 to 72.536636, 50 fc rgb "#0000FF"
set object 54 rect from 0.068185, 40 to 73.485204, 50 fc rgb "#FFFF00"
set object 55 rect from 0.069065, 40 to 74.764211, 50 fc rgb "#FF00FF"
set object 56 rect from 0.070262, 40 to 75.200140, 50 fc rgb "#808080"
set object 57 rect from 0.070696, 40 to 75.865214, 50 fc rgb "#800080"
set object 58 rect from 0.071598, 40 to 76.551602, 50 fc rgb "#008080"
set object 59 rect from 0.071944, 40 to 147.370125, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.137107, 50 to 152.281469, 60 fc rgb "#FF0000"
set object 61 rect from 0.019668, 50 to 47.284125, 60 fc rgb "#00FF00"
set object 62 rect from 0.044962, 50 to 48.716584, 60 fc rgb "#0000FF"
set object 63 rect from 0.045849, 50 to 51.036879, 60 fc rgb "#FFFF00"
set object 64 rect from 0.048033, 50 to 52.966031, 60 fc rgb "#FF00FF"
set object 65 rect from 0.049849, 50 to 53.743010, 60 fc rgb "#808080"
set object 66 rect from 0.050581, 50 to 54.565823, 60 fc rgb "#800080"
set object 67 rect from 0.051905, 50 to 55.879962, 60 fc rgb "#008080"
set object 68 rect from 0.052565, 50 to 145.110564, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
