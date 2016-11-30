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

set object 15 rect from 0.956684, 0 to 176.417834, 10 fc rgb "#FF0000"
set object 16 rect from 0.725976, 0 to 131.051754, 10 fc rgb "#00FF00"
set object 17 rect from 0.738976, 0 to 131.781094, 10 fc rgb "#0000FF"
set object 18 rect from 0.741570, 0 to 132.503850, 10 fc rgb "#FFFF00"
set object 19 rect from 0.745907, 0 to 133.793177, 10 fc rgb "#FF00FF"
set object 20 rect from 0.753010, 0 to 137.549803, 10 fc rgb "#808080"
set object 21 rect from 0.774021, 0 to 138.169739, 10 fc rgb "#800080"
set object 22 rect from 0.779237, 0 to 138.811202, 10 fc rgb "#008080"
set object 23 rect from 0.781100, 0 to 169.469199, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.833682, 10 to 156.411733, 20 fc rgb "#FF0000"
set object 25 rect from 0.143189, 10 to 28.590768, 20 fc rgb "#00FF00"
set object 26 rect from 0.163895, 10 to 29.821393, 20 fc rgb "#0000FF"
set object 27 rect from 0.168483, 10 to 31.390181, 20 fc rgb "#FFFF00"
set object 28 rect from 0.177425, 10 to 33.110175, 20 fc rgb "#FF00FF"
set object 29 rect from 0.187059, 10 to 37.040773, 20 fc rgb "#808080"
set object 30 rect from 0.209143, 10 to 37.779362, 20 fc rgb "#800080"
set object 31 rect from 0.216026, 10 to 38.715228, 20 fc rgb "#008080"
set object 32 rect from 0.218485, 10 to 147.521278, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.821609, 20 to 154.780324, 30 fc rgb "#FF0000"
set object 34 rect from 0.275905, 20 to 51.217329, 30 fc rgb "#00FF00"
set object 35 rect from 0.290331, 20 to 52.047530, 30 fc rgb "#0000FF"
set object 36 rect from 0.293418, 20 to 54.386036, 30 fc rgb "#FFFF00"
set object 37 rect from 0.306753, 20 to 56.365033, 30 fc rgb "#FF00FF"
set object 38 rect from 0.317816, 20 to 60.024888, 30 fc rgb "#808080"
set object 39 rect from 0.338402, 20 to 60.805814, 30 fc rgb "#800080"
set object 40 rect from 0.344449, 20 to 61.965463, 30 fc rgb "#008080"
set object 41 rect from 0.349224, 20 to 145.132786, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.933321, 30 to 173.814628, 40 fc rgb "#FF0000"
set object 43 rect from 0.515777, 30 to 93.371654, 40 fc rgb "#00FF00"
set object 44 rect from 0.527297, 30 to 94.047269, 40 fc rgb "#0000FF"
set object 45 rect from 0.529444, 30 to 96.266059, 40 fc rgb "#FFFF00"
set object 46 rect from 0.541987, 30 to 97.734698, 40 fc rgb "#FF00FF"
set object 47 rect from 0.550170, 30 to 101.321796, 40 fc rgb "#808080"
set object 48 rect from 0.570347, 30 to 102.005238, 40 fc rgb "#800080"
set object 49 rect from 0.576061, 30 to 102.745428, 40 fc rgb "#008080"
set object 50 rect from 0.578361, 30 to 165.432047, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.944963, 40 to 175.734039, 50 fc rgb "#FF0000"
set object 52 rect from 0.619669, 40 to 111.745997, 50 fc rgb "#00FF00"
set object 53 rect from 0.630593, 40 to 112.398309, 50 fc rgb "#0000FF"
set object 54 rect from 0.632726, 40 to 114.624568, 50 fc rgb "#FFFF00"
set object 55 rect from 0.645119, 40 to 116.058519, 50 fc rgb "#FF00FF"
set object 56 rect from 0.653181, 40 to 119.589229, 50 fc rgb "#808080"
set object 57 rect from 0.673038, 40 to 120.238161, 50 fc rgb "#800080"
set object 58 rect from 0.678497, 40 to 120.951133, 50 fc rgb "#008080"
set object 59 rect from 0.680698, 40 to 167.510661, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.913639, 50 to 170.678475, 60 fc rgb "#FF0000"
set object 61 rect from 0.402178, 50 to 73.401484, 60 fc rgb "#00FF00"
set object 62 rect from 0.415120, 50 to 74.205715, 60 fc rgb "#0000FF"
set object 63 rect from 0.418090, 50 to 76.430906, 60 fc rgb "#FFFF00"
set object 64 rect from 0.430442, 50 to 78.158192, 60 fc rgb "#FF00FF"
set object 65 rect from 0.440129, 50 to 81.735509, 60 fc rgb "#808080"
set object 66 rect from 0.460261, 50 to 82.442254, 60 fc rgb "#800080"
set object 67 rect from 0.465930, 50 to 83.144021, 60 fc rgb "#008080"
set object 68 rect from 0.468181, 50 to 161.645890, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
