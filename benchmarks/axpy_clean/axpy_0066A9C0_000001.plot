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

set object 15 rect from 0.116229, 0 to 162.425256, 10 fc rgb "#FF0000"
set object 16 rect from 0.096247, 0 to 132.805958, 10 fc rgb "#00FF00"
set object 17 rect from 0.098053, 0 to 133.636617, 10 fc rgb "#0000FF"
set object 18 rect from 0.098419, 0 to 134.612728, 10 fc rgb "#FFFF00"
set object 19 rect from 0.099138, 0 to 136.014338, 10 fc rgb "#FF00FF"
set object 20 rect from 0.100167, 0 to 137.440461, 10 fc rgb "#808080"
set object 21 rect from 0.101231, 0 to 138.222176, 10 fc rgb "#800080"
set object 22 rect from 0.102074, 0 to 139.003892, 10 fc rgb "#008080"
set object 23 rect from 0.102369, 0 to 157.211536, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.114669, 10 to 160.767990, 20 fc rgb "#FF0000"
set object 25 rect from 0.081623, 10 to 113.223849, 20 fc rgb "#00FF00"
set object 26 rect from 0.083649, 10 to 114.151017, 20 fc rgb "#0000FF"
set object 27 rect from 0.084086, 10 to 115.196451, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084885, 10 to 116.834634, 20 fc rgb "#FF00FF"
set object 29 rect from 0.086083, 10 to 118.312414, 20 fc rgb "#808080"
set object 30 rect from 0.087173, 10 to 119.194731, 20 fc rgb "#800080"
set object 31 rect from 0.088056, 10 to 120.010440, 20 fc rgb "#008080"
set object 32 rect from 0.088402, 10 to 155.006453, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.112919, 20 to 157.838278, 30 fc rgb "#FF0000"
set object 34 rect from 0.068892, 20 to 95.651049, 30 fc rgb "#00FF00"
set object 35 rect from 0.070740, 20 to 96.503465, 30 fc rgb "#0000FF"
set object 36 rect from 0.071109, 20 to 97.523091, 30 fc rgb "#FFFF00"
set object 37 rect from 0.071857, 20 to 99.004923, 30 fc rgb "#FF00FF"
set object 38 rect from 0.072941, 20 to 100.206708, 30 fc rgb "#808080"
set object 39 rect from 0.073869, 20 to 101.022417, 30 fc rgb "#800080"
set object 40 rect from 0.074715, 20 to 101.878884, 30 fc rgb "#008080"
set object 41 rect from 0.075059, 20 to 152.859794, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.111415, 30 to 156.118455, 40 fc rgb "#FF0000"
set object 43 rect from 0.053561, 30 to 75.012607, 40 fc rgb "#00FF00"
set object 44 rect from 0.055562, 30 to 75.879974, 40 fc rgb "#0000FF"
set object 45 rect from 0.055934, 30 to 77.055950, 40 fc rgb "#FFFF00"
set object 46 rect from 0.056799, 30 to 78.646527, 40 fc rgb "#FF00FF"
set object 47 rect from 0.057972, 30 to 79.867355, 40 fc rgb "#808080"
set object 48 rect from 0.058892, 30 to 80.673542, 40 fc rgb "#800080"
set object 49 rect from 0.059727, 30 to 81.550430, 40 fc rgb "#008080"
set object 50 rect from 0.060106, 30 to 150.645190, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.109755, 40 to 153.982695, 50 fc rgb "#FF0000"
set object 52 rect from 0.039278, 40 to 56.030054, 50 fc rgb "#00FF00"
set object 53 rect from 0.041609, 40 to 56.878378, 50 fc rgb "#0000FF"
set object 54 rect from 0.041958, 40 to 57.956427, 50 fc rgb "#FFFF00"
set object 55 rect from 0.042783, 40 to 59.755054, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044091, 40 to 61.064247, 50 fc rgb "#808080"
set object 57 rect from 0.045061, 40 to 61.911193, 50 fc rgb "#800080"
set object 58 rect from 0.045926, 40 to 62.839739, 50 fc rgb "#008080"
set object 59 rect from 0.046370, 40 to 148.226669, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.107939, 50 to 154.215257, 60 fc rgb "#FF0000"
set object 61 rect from 0.020519, 50 to 31.862385, 60 fc rgb "#00FF00"
set object 62 rect from 0.023896, 50 to 33.330644, 60 fc rgb "#0000FF"
set object 63 rect from 0.024651, 50 to 35.920467, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026584, 50 to 38.126928, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028176, 50 to 39.573389, 60 fc rgb "#808080"
set object 66 rect from 0.029260, 50 to 40.563114, 60 fc rgb "#800080"
set object 67 rect from 0.030395, 50 to 42.144210, 60 fc rgb "#008080"
set object 68 rect from 0.031128, 50 to 145.335043, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
