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

set object 15 rect from 0.124955, 0 to 161.989347, 10 fc rgb "#FF0000"
set object 16 rect from 0.103462, 0 to 132.117406, 10 fc rgb "#00FF00"
set object 17 rect from 0.105757, 0 to 132.821838, 10 fc rgb "#0000FF"
set object 18 rect from 0.106079, 0 to 133.739355, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106824, 0 to 135.031649, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107842, 0 to 137.721528, 10 fc rgb "#808080"
set object 21 rect from 0.109989, 0 to 138.369556, 10 fc rgb "#800080"
set object 22 rect from 0.110756, 0 to 139.032624, 10 fc rgb "#008080"
set object 23 rect from 0.111059, 0 to 155.989137, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.123438, 10 to 160.661955, 20 fc rgb "#FF0000"
set object 25 rect from 0.087605, 10 to 112.262944, 20 fc rgb "#00FF00"
set object 26 rect from 0.089983, 10 to 113.209289, 20 fc rgb "#0000FF"
set object 27 rect from 0.090443, 10 to 114.308555, 20 fc rgb "#FFFF00"
set object 28 rect from 0.091342, 10 to 115.822708, 20 fc rgb "#FF00FF"
set object 29 rect from 0.092551, 10 to 118.530135, 20 fc rgb "#808080"
set object 30 rect from 0.094705, 10 to 119.287211, 20 fc rgb "#800080"
set object 31 rect from 0.095554, 10 to 120.014206, 20 fc rgb "#008080"
set object 32 rect from 0.095866, 10 to 153.928484, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121693, 20 to 159.576476, 30 fc rgb "#FF0000"
set object 34 rect from 0.071467, 20 to 91.818112, 30 fc rgb "#00FF00"
set object 35 rect from 0.073615, 20 to 92.566415, 30 fc rgb "#0000FF"
set object 36 rect from 0.073964, 20 to 95.011873, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075911, 20 to 96.473382, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077081, 20 to 98.966471, 30 fc rgb "#808080"
set object 39 rect from 0.079082, 20 to 99.753630, 30 fc rgb "#800080"
set object 40 rect from 0.079945, 20 to 100.598448, 30 fc rgb "#008080"
set object 41 rect from 0.080374, 20 to 151.964347, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.120234, 30 to 157.954527, 40 fc rgb "#FF0000"
set object 43 rect from 0.054837, 30 to 71.207827, 40 fc rgb "#00FF00"
set object 44 rect from 0.057166, 30 to 71.951116, 40 fc rgb "#0000FF"
set object 45 rect from 0.057515, 30 to 74.415376, 40 fc rgb "#FFFF00"
set object 46 rect from 0.059482, 30 to 75.944571, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060719, 30 to 78.531667, 40 fc rgb "#808080"
set object 48 rect from 0.062790, 30 to 79.381498, 40 fc rgb "#800080"
set object 49 rect from 0.063695, 30 to 80.266426, 40 fc rgb "#008080"
set object 50 rect from 0.064154, 30 to 149.960098, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.118656, 40 to 156.074372, 50 fc rgb "#FF0000"
set object 52 rect from 0.038042, 40 to 50.373177, 50 fc rgb "#00FF00"
set object 53 rect from 0.040604, 40 to 51.179138, 50 fc rgb "#0000FF"
set object 54 rect from 0.040945, 40 to 53.722364, 50 fc rgb "#FFFF00"
set object 55 rect from 0.042975, 40 to 55.353087, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044295, 40 to 57.896313, 50 fc rgb "#808080"
set object 57 rect from 0.046318, 40 to 58.714809, 50 fc rgb "#800080"
set object 58 rect from 0.047253, 40 to 59.671182, 50 fc rgb "#008080"
set object 59 rect from 0.047737, 40 to 147.890673, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116798, 50 to 158.294212, 60 fc rgb "#FF0000"
set object 61 rect from 0.015216, 50 to 22.484175, 60 fc rgb "#00FF00"
set object 62 rect from 0.018480, 50 to 23.978275, 60 fc rgb "#0000FF"
set object 63 rect from 0.019271, 50 to 29.607465, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023791, 50 to 31.802236, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025499, 50 to 34.676370, 60 fc rgb "#808080"
set object 66 rect from 0.027834, 50 to 35.659066, 60 fc rgb "#800080"
set object 67 rect from 0.029013, 50 to 37.170711, 60 fc rgb "#008080"
set object 68 rect from 0.029792, 50 to 145.346193, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
