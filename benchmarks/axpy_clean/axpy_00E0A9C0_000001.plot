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

set object 15 rect from 0.141726, 0 to 159.369935, 10 fc rgb "#FF0000"
set object 16 rect from 0.119702, 0 to 133.218906, 10 fc rgb "#00FF00"
set object 17 rect from 0.121915, 0 to 134.160347, 10 fc rgb "#0000FF"
set object 18 rect from 0.122526, 0 to 135.025070, 10 fc rgb "#FFFF00"
set object 19 rect from 0.123315, 0 to 136.153924, 10 fc rgb "#FF00FF"
set object 20 rect from 0.124342, 0 to 137.483340, 10 fc rgb "#808080"
set object 21 rect from 0.125564, 0 to 138.035710, 10 fc rgb "#800080"
set object 22 rect from 0.126317, 0 to 138.594658, 10 fc rgb "#008080"
set object 23 rect from 0.126573, 0 to 154.692321, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.140133, 10 to 157.881605, 20 fc rgb "#FF0000"
set object 25 rect from 0.103367, 10 to 115.319462, 20 fc rgb "#00FF00"
set object 26 rect from 0.105564, 10 to 116.153497, 20 fc rgb "#0000FF"
set object 27 rect from 0.106094, 10 to 117.018220, 20 fc rgb "#FFFF00"
set object 28 rect from 0.106915, 10 to 118.354213, 20 fc rgb "#FF00FF"
set object 29 rect from 0.108136, 10 to 119.812954, 20 fc rgb "#808080"
set object 30 rect from 0.109468, 10 to 120.472731, 20 fc rgb "#800080"
set object 31 rect from 0.110309, 10 to 121.163194, 20 fc rgb "#008080"
set object 32 rect from 0.110672, 10 to 152.878484, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.136836, 20 to 154.242970, 30 fc rgb "#FF0000"
set object 34 rect from 0.072448, 20 to 81.545932, 30 fc rgb "#00FF00"
set object 35 rect from 0.074784, 20 to 82.282427, 30 fc rgb "#0000FF"
set object 36 rect from 0.075191, 20 to 83.246884, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076069, 20 to 84.622331, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077325, 20 to 85.889276, 30 fc rgb "#808080"
set object 39 rect from 0.078483, 20 to 86.588508, 30 fc rgb "#800080"
set object 40 rect from 0.079372, 20 to 87.261436, 30 fc rgb "#008080"
set object 41 rect from 0.079736, 20 to 149.317666, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.135240, 30 to 152.829166, 40 fc rgb "#FF0000"
set object 43 rect from 0.056668, 30 to 64.411478, 40 fc rgb "#00FF00"
set object 44 rect from 0.059167, 30 to 65.305792, 40 fc rgb "#0000FF"
set object 45 rect from 0.059706, 30 to 66.403957, 40 fc rgb "#FFFF00"
set object 46 rect from 0.060705, 30 to 67.872562, 40 fc rgb "#FF00FF"
set object 47 rect from 0.062065, 30 to 69.173484, 40 fc rgb "#808080"
set object 48 rect from 0.063242, 30 to 69.879291, 40 fc rgb "#800080"
set object 49 rect from 0.064170, 30 to 70.830596, 40 fc rgb "#008080"
set object 50 rect from 0.064776, 30 to 147.567395, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.133457, 40 to 154.191462, 50 fc rgb "#FF0000"
set object 52 rect from 0.034368, 40 to 40.959829, 50 fc rgb "#00FF00"
set object 53 rect from 0.037843, 40 to 42.282670, 50 fc rgb "#0000FF"
set object 54 rect from 0.038710, 40 to 45.041236, 50 fc rgb "#FFFF00"
set object 55 rect from 0.041253, 40 to 47.249624, 50 fc rgb "#FF00FF"
set object 56 rect from 0.043264, 40 to 49.009757, 50 fc rgb "#808080"
set object 57 rect from 0.044854, 40 to 49.885441, 50 fc rgb "#800080"
set object 58 rect from 0.046070, 40 to 51.489946, 50 fc rgb "#008080"
set object 59 rect from 0.047107, 40 to 145.167212, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.138405, 50 to 155.768567, 60 fc rgb "#FF0000"
set object 61 rect from 0.088865, 50 to 99.322628, 60 fc rgb "#00FF00"
set object 62 rect from 0.090968, 50 to 99.949524, 60 fc rgb "#0000FF"
set object 63 rect from 0.091311, 50 to 100.973164, 60 fc rgb "#FFFF00"
set object 64 rect from 0.092243, 50 to 102.200655, 60 fc rgb "#FF00FF"
set object 65 rect from 0.093362, 50 to 103.466505, 60 fc rgb "#808080"
set object 66 rect from 0.094519, 50 to 104.147104, 60 fc rgb "#800080"
set object 67 rect from 0.095397, 50 to 104.804688, 60 fc rgb "#008080"
set object 68 rect from 0.095743, 50 to 151.121639, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
