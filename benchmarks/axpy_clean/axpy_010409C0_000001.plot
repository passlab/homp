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

set object 15 rect from 0.135660, 0 to 156.792341, 10 fc rgb "#FF0000"
set object 16 rect from 0.084734, 0 to 97.598199, 10 fc rgb "#00FF00"
set object 17 rect from 0.087099, 0 to 98.362406, 10 fc rgb "#0000FF"
set object 18 rect from 0.087508, 0 to 99.227910, 10 fc rgb "#FFFF00"
set object 19 rect from 0.088299, 0 to 100.474958, 10 fc rgb "#FF00FF"
set object 20 rect from 0.089397, 0 to 101.940349, 10 fc rgb "#808080"
set object 21 rect from 0.090713, 0 to 102.585254, 10 fc rgb "#800080"
set object 22 rect from 0.091507, 0 to 103.175011, 10 fc rgb "#008080"
set object 23 rect from 0.091805, 0 to 151.962859, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.132104, 10 to 153.501411, 20 fc rgb "#FF0000"
set object 25 rect from 0.050420, 10 to 59.361845, 20 fc rgb "#00FF00"
set object 26 rect from 0.053122, 10 to 60.449072, 20 fc rgb "#0000FF"
set object 27 rect from 0.053823, 10 to 61.447383, 20 fc rgb "#FFFF00"
set object 28 rect from 0.054746, 10 to 62.855375, 20 fc rgb "#FF00FF"
set object 29 rect from 0.055988, 10 to 64.401800, 20 fc rgb "#808080"
set object 30 rect from 0.057375, 10 to 65.109735, 20 fc rgb "#800080"
set object 31 rect from 0.058248, 10 to 65.801914, 20 fc rgb "#008080"
set object 32 rect from 0.058595, 10 to 147.835678, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.130024, 20 to 154.054020, 30 fc rgb "#FF0000"
set object 34 rect from 0.028024, 20 to 35.299941, 30 fc rgb "#00FF00"
set object 35 rect from 0.031798, 20 to 36.765332, 30 fc rgb "#0000FF"
set object 36 rect from 0.032800, 20 to 39.138998, 30 fc rgb "#FFFF00"
set object 37 rect from 0.034937, 20 to 41.489026, 30 fc rgb "#FF00FF"
set object 38 rect from 0.036993, 20 to 43.107483, 30 fc rgb "#808080"
set object 39 rect from 0.038450, 20 to 44.002248, 30 fc rgb "#800080"
set object 40 rect from 0.039724, 20 to 45.442881, 30 fc rgb "#008080"
set object 41 rect from 0.040511, 20 to 145.263928, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133859, 30 to 155.302195, 40 fc rgb "#FF0000"
set object 43 rect from 0.067457, 30 to 77.952729, 40 fc rgb "#00FF00"
set object 44 rect from 0.069744, 30 to 78.871129, 40 fc rgb "#0000FF"
set object 45 rect from 0.070206, 30 to 79.886324, 40 fc rgb "#FFFF00"
set object 46 rect from 0.071129, 30 to 81.474392, 40 fc rgb "#FF00FF"
set object 47 rect from 0.072544, 30 to 82.886887, 40 fc rgb "#808080"
set object 48 rect from 0.073786, 30 to 83.661224, 40 fc rgb "#800080"
set object 49 rect from 0.074701, 30 to 84.465953, 40 fc rgb "#008080"
set object 50 rect from 0.075170, 30 to 149.917836, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.138656, 40 to 160.088906, 50 fc rgb "#FF0000"
set object 52 rect from 0.115997, 40 to 132.476082, 50 fc rgb "#00FF00"
set object 53 rect from 0.118069, 40 to 133.218905, 50 fc rgb "#0000FF"
set object 54 rect from 0.118477, 40 to 134.109172, 50 fc rgb "#FFFF00"
set object 55 rect from 0.119270, 40 to 135.329205, 50 fc rgb "#FF00FF"
set object 56 rect from 0.120352, 40 to 136.594257, 50 fc rgb "#808080"
set object 57 rect from 0.121487, 40 to 137.313450, 50 fc rgb "#800080"
set object 58 rect from 0.122368, 40 to 137.997746, 50 fc rgb "#008080"
set object 59 rect from 0.122725, 40 to 155.546425, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.137185, 50 to 158.907140, 60 fc rgb "#FF0000"
set object 61 rect from 0.099949, 50 to 114.467077, 60 fc rgb "#00FF00"
set object 62 rect from 0.102076, 50 to 115.194148, 60 fc rgb "#0000FF"
set object 63 rect from 0.102463, 50 to 116.183452, 60 fc rgb "#FFFF00"
set object 64 rect from 0.103367, 50 to 117.778275, 60 fc rgb "#FF00FF"
set object 65 rect from 0.104761, 50 to 119.077095, 60 fc rgb "#808080"
set object 66 rect from 0.105924, 50 to 119.819918, 60 fc rgb "#800080"
set object 67 rect from 0.106826, 50 to 120.522227, 60 fc rgb "#008080"
set object 68 rect from 0.107199, 50 to 153.854812, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
