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

set object 15 rect from 0.123515, 0 to 153.366172, 10 fc rgb "#FF0000"
set object 16 rect from 0.044537, 0 to 56.471674, 10 fc rgb "#00FF00"
set object 17 rect from 0.047211, 0 to 57.264545, 10 fc rgb "#0000FF"
set object 18 rect from 0.047640, 0 to 58.179116, 10 fc rgb "#FFFF00"
set object 19 rect from 0.048409, 0 to 59.492511, 10 fc rgb "#FF00FF"
set object 20 rect from 0.049505, 0 to 61.245739, 10 fc rgb "#808080"
set object 21 rect from 0.050952, 0 to 61.892813, 10 fc rgb "#800080"
set object 22 rect from 0.051758, 0 to 62.550695, 10 fc rgb "#008080"
set object 23 rect from 0.052027, 0 to 148.052298, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.121626, 10 to 153.737308, 20 fc rgb "#FF0000"
set object 25 rect from 0.024343, 10 to 32.695397, 20 fc rgb "#00FF00"
set object 26 rect from 0.027597, 10 to 34.053393, 20 fc rgb "#0000FF"
set object 27 rect from 0.028404, 10 to 36.205445, 20 fc rgb "#FFFF00"
set object 28 rect from 0.030207, 10 to 38.162322, 20 fc rgb "#FF00FF"
set object 29 rect from 0.031802, 10 to 40.145665, 20 fc rgb "#808080"
set object 30 rect from 0.033466, 10 to 40.985543, 20 fc rgb "#800080"
set object 31 rect from 0.034596, 10 to 41.924211, 20 fc rgb "#008080"
set object 32 rect from 0.034923, 10 to 145.435132, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129844, 20 to 161.132094, 30 fc rgb "#FF0000"
set object 34 rect from 0.107738, 20 to 131.950407, 30 fc rgb "#00FF00"
set object 35 rect from 0.109854, 20 to 132.642046, 30 fc rgb "#0000FF"
set object 36 rect from 0.110212, 20 to 133.692791, 30 fc rgb "#FFFF00"
set object 37 rect from 0.111069, 20 to 135.324282, 30 fc rgb "#FF00FF"
set object 38 rect from 0.112420, 20 to 136.743721, 30 fc rgb "#808080"
set object 39 rect from 0.113610, 20 to 137.495617, 30 fc rgb "#800080"
set object 40 rect from 0.114483, 20 to 138.241517, 30 fc rgb "#008080"
set object 41 rect from 0.114845, 20 to 155.867669, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.128367, 30 to 159.518738, 40 fc rgb "#FF0000"
set object 43 rect from 0.092650, 30 to 113.607299, 40 fc rgb "#00FF00"
set object 44 rect from 0.094650, 30 to 114.535122, 40 fc rgb "#0000FF"
set object 45 rect from 0.095167, 30 to 115.682251, 40 fc rgb "#FFFF00"
set object 46 rect from 0.096132, 30 to 117.169166, 40 fc rgb "#FF00FF"
set object 47 rect from 0.097352, 30 to 118.571727, 40 fc rgb "#808080"
set object 48 rect from 0.098530, 30 to 119.334469, 40 fc rgb "#800080"
set object 49 rect from 0.099411, 30 to 120.076741, 40 fc rgb "#008080"
set object 50 rect from 0.099773, 30 to 154.103596, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126862, 40 to 157.616121, 50 fc rgb "#FF0000"
set object 52 rect from 0.077986, 40 to 96.167918, 50 fc rgb "#00FF00"
set object 53 rect from 0.080182, 40 to 96.970413, 50 fc rgb "#0000FF"
set object 54 rect from 0.080590, 40 to 98.050065, 50 fc rgb "#FFFF00"
set object 55 rect from 0.081487, 40 to 99.555043, 50 fc rgb "#FF00FF"
set object 56 rect from 0.082737, 40 to 101.005832, 50 fc rgb "#808080"
set object 57 rect from 0.083954, 40 to 101.801109, 50 fc rgb "#800080"
set object 58 rect from 0.084881, 40 to 102.562630, 50 fc rgb "#008080"
set object 59 rect from 0.085241, 40 to 152.147940, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.125182, 50 to 156.402666, 60 fc rgb "#FF0000"
set object 61 rect from 0.060749, 50 to 75.616092, 60 fc rgb "#00FF00"
set object 62 rect from 0.063196, 50 to 76.633080, 60 fc rgb "#0000FF"
set object 63 rect from 0.063710, 50 to 77.948917, 60 fc rgb "#FFFF00"
set object 64 rect from 0.064839, 50 to 79.774398, 60 fc rgb "#FF00FF"
set object 65 rect from 0.066357, 50 to 81.319165, 60 fc rgb "#808080"
set object 66 rect from 0.067635, 50 to 82.218079, 60 fc rgb "#800080"
set object 67 rect from 0.068621, 50 to 83.421839, 60 fc rgb "#008080"
set object 68 rect from 0.069348, 50 to 150.076615, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
