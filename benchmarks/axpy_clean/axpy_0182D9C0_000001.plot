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

set object 15 rect from 0.127654, 0 to 162.644160, 10 fc rgb "#FF0000"
set object 16 rect from 0.104392, 0 to 131.996186, 10 fc rgb "#00FF00"
set object 17 rect from 0.106994, 0 to 132.793714, 10 fc rgb "#0000FF"
set object 18 rect from 0.107353, 0 to 133.767105, 10 fc rgb "#FFFF00"
set object 19 rect from 0.108143, 0 to 135.185080, 10 fc rgb "#FF00FF"
set object 20 rect from 0.109284, 0 to 136.834612, 10 fc rgb "#808080"
set object 21 rect from 0.110619, 0 to 137.521935, 10 fc rgb "#800080"
set object 22 rect from 0.111452, 0 to 138.267460, 10 fc rgb "#008080"
set object 23 rect from 0.111783, 0 to 157.274418, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.125857, 10 to 160.891805, 20 fc rgb "#FF0000"
set object 25 rect from 0.087861, 10 to 111.622037, 20 fc rgb "#00FF00"
set object 26 rect from 0.090507, 10 to 112.492605, 20 fc rgb "#0000FF"
set object 27 rect from 0.090965, 10 to 113.515525, 20 fc rgb "#FFFF00"
set object 28 rect from 0.091819, 10 to 115.122982, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093105, 10 to 116.835698, 20 fc rgb "#808080"
set object 30 rect from 0.094495, 10 to 117.677810, 20 fc rgb "#800080"
set object 31 rect from 0.095430, 10 to 118.493903, 20 fc rgb "#008080"
set object 32 rect from 0.095818, 10 to 155.031678, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.123953, 20 to 158.403816, 30 fc rgb "#FF0000"
set object 34 rect from 0.073054, 20 to 92.960568, 30 fc rgb "#00FF00"
set object 35 rect from 0.075447, 20 to 93.846009, 30 fc rgb "#0000FF"
set object 36 rect from 0.075930, 20 to 95.002681, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076839, 20 to 96.486277, 30 fc rgb "#FF00FF"
set object 38 rect from 0.078043, 20 to 98.036749, 30 fc rgb "#808080"
set object 39 rect from 0.079308, 20 to 98.862770, 30 fc rgb "#800080"
set object 40 rect from 0.080239, 20 to 99.709827, 30 fc rgb "#008080"
set object 41 rect from 0.080648, 20 to 152.852085, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.122219, 30 to 156.687409, 40 fc rgb "#FF0000"
set object 43 rect from 0.055434, 30 to 71.314559, 40 fc rgb "#00FF00"
set object 44 rect from 0.057994, 30 to 72.185164, 40 fc rgb "#0000FF"
set object 45 rect from 0.058415, 30 to 73.476805, 40 fc rgb "#FFFF00"
set object 46 rect from 0.059456, 30 to 75.118918, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060783, 30 to 76.738739, 40 fc rgb "#808080"
set object 48 rect from 0.062106, 30 to 77.609343, 40 fc rgb "#800080"
set object 49 rect from 0.063121, 30 to 78.496039, 40 fc rgb "#008080"
set object 50 rect from 0.063522, 30 to 150.626691, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.120425, 40 to 154.428577, 50 fc rgb "#FF0000"
set object 52 rect from 0.038715, 40 to 51.097672, 50 fc rgb "#00FF00"
set object 53 rect from 0.041657, 40 to 51.967021, 50 fc rgb "#0000FF"
set object 54 rect from 0.042093, 40 to 53.169495, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043068, 40 to 54.950342, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044517, 40 to 56.572636, 50 fc rgb "#808080"
set object 57 rect from 0.045832, 40 to 57.425894, 50 fc rgb "#800080"
set object 58 rect from 0.046781, 40 to 58.279152, 50 fc rgb "#008080"
set object 59 rect from 0.047213, 40 to 148.265072, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.118449, 50 to 154.935091, 60 fc rgb "#FF0000"
set object 61 rect from 0.016954, 50 to 24.545107, 60 fc rgb "#00FF00"
set object 62 rect from 0.020288, 50 to 26.349464, 60 fc rgb "#0000FF"
set object 63 rect from 0.021420, 50 to 28.673954, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023329, 50 to 30.914223, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025102, 50 to 32.924153, 60 fc rgb "#808080"
set object 66 rect from 0.026734, 50 to 33.949546, 60 fc rgb "#800080"
set object 67 rect from 0.028158, 50 to 35.848016, 60 fc rgb "#008080"
set object 68 rect from 0.029103, 50 to 145.603729, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
