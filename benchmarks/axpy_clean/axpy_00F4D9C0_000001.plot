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

set object 15 rect from 0.140939, 0 to 158.993282, 10 fc rgb "#FF0000"
set object 16 rect from 0.119501, 0 to 133.551804, 10 fc rgb "#00FF00"
set object 17 rect from 0.121695, 0 to 134.315555, 10 fc rgb "#0000FF"
set object 18 rect from 0.122163, 0 to 135.156340, 10 fc rgb "#FFFF00"
set object 19 rect from 0.122934, 0 to 136.254645, 10 fc rgb "#FF00FF"
set object 20 rect from 0.123937, 0 to 137.658889, 10 fc rgb "#808080"
set object 21 rect from 0.125209, 0 to 138.221247, 10 fc rgb "#800080"
set object 22 rect from 0.125962, 0 to 138.780304, 10 fc rgb "#008080"
set object 23 rect from 0.126220, 0 to 154.483412, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.139374, 10 to 157.478987, 20 fc rgb "#FF0000"
set object 25 rect from 0.104298, 10 to 116.782306, 20 fc rgb "#00FF00"
set object 26 rect from 0.106486, 10 to 117.546057, 20 fc rgb "#0000FF"
set object 27 rect from 0.106922, 10 to 118.402250, 20 fc rgb "#FFFF00"
set object 28 rect from 0.107729, 10 to 119.697544, 20 fc rgb "#FF00FF"
set object 29 rect from 0.108906, 10 to 121.087482, 20 fc rgb "#808080"
set object 30 rect from 0.110164, 10 to 121.738981, 20 fc rgb "#800080"
set object 31 rect from 0.111002, 10 to 122.381676, 20 fc rgb "#008080"
set object 32 rect from 0.111333, 10 to 152.687388, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137698, 20 to 155.543198, 30 fc rgb "#FF0000"
set object 34 rect from 0.089551, 20 to 100.519041, 30 fc rgb "#00FF00"
set object 35 rect from 0.091727, 20 to 101.181545, 30 fc rgb "#0000FF"
set object 36 rect from 0.092052, 20 to 102.129080, 30 fc rgb "#FFFF00"
set object 37 rect from 0.092914, 20 to 103.371550, 30 fc rgb "#FF00FF"
set object 38 rect from 0.094059, 20 to 104.652538, 30 fc rgb "#808080"
set object 39 rect from 0.095221, 20 to 105.355761, 30 fc rgb "#800080"
set object 40 rect from 0.096139, 20 to 106.040275, 30 fc rgb "#008080"
set object 41 rect from 0.096478, 20 to 151.031127, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.136271, 30 to 154.032205, 40 fc rgb "#FF0000"
set object 43 rect from 0.072533, 30 to 81.702603, 40 fc rgb "#00FF00"
set object 44 rect from 0.074670, 30 to 82.485063, 40 fc rgb "#0000FF"
set object 45 rect from 0.075095, 30 to 83.496427, 40 fc rgb "#FFFF00"
set object 46 rect from 0.075997, 30 to 84.731194, 40 fc rgb "#FF00FF"
set object 47 rect from 0.077119, 30 to 86.001177, 40 fc rgb "#808080"
set object 48 rect from 0.078284, 30 to 86.704400, 40 fc rgb "#800080"
set object 49 rect from 0.079175, 30 to 87.364703, 40 fc rgb "#008080"
set object 50 rect from 0.079522, 30 to 149.345154, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.134653, 40 to 152.359438, 50 fc rgb "#FF0000"
set object 52 rect from 0.056662, 40 to 64.657980, 50 fc rgb "#00FF00"
set object 53 rect from 0.059129, 40 to 65.386514, 50 fc rgb "#0000FF"
set object 54 rect from 0.059528, 40 to 66.323044, 50 fc rgb "#FFFF00"
set object 55 rect from 0.060388, 40 to 67.685469, 50 fc rgb "#FF00FF"
set object 56 rect from 0.061631, 40 to 68.969759, 50 fc rgb "#808080"
set object 57 rect from 0.062808, 40 to 69.670781, 50 fc rgb "#800080"
set object 58 rect from 0.063725, 40 to 70.456541, 50 fc rgb "#008080"
set object 59 rect from 0.064170, 40 to 147.426973, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.132859, 50 to 153.218932, 60 fc rgb "#FF0000"
set object 61 rect from 0.034733, 50 to 41.650819, 60 fc rgb "#00FF00"
set object 62 rect from 0.038471, 50 to 43.108988, 60 fc rgb "#0000FF"
set object 63 rect from 0.039312, 50 to 45.530099, 60 fc rgb "#FFFF00"
set object 64 rect from 0.041521, 50 to 47.426270, 60 fc rgb "#FF00FF"
set object 65 rect from 0.043224, 50 to 49.009897, 60 fc rgb "#808080"
set object 66 rect from 0.044679, 50 to 49.820969, 60 fc rgb "#800080"
set object 67 rect from 0.045813, 50 to 51.066741, 60 fc rgb "#008080"
set object 68 rect from 0.046536, 50 to 145.199551, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
