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

set object 15 rect from 0.142099, 0 to 158.875911, 10 fc rgb "#FF0000"
set object 16 rect from 0.121615, 0 to 134.091072, 10 fc rgb "#00FF00"
set object 17 rect from 0.123578, 0 to 134.758194, 10 fc rgb "#0000FF"
set object 18 rect from 0.123944, 0 to 135.448173, 10 fc rgb "#FFFF00"
set object 19 rect from 0.124583, 0 to 136.586532, 10 fc rgb "#FF00FF"
set object 20 rect from 0.125624, 0 to 138.526961, 10 fc rgb "#808080"
set object 21 rect from 0.127411, 0 to 139.116817, 10 fc rgb "#800080"
set object 22 rect from 0.128208, 0 to 139.693611, 10 fc rgb "#008080"
set object 23 rect from 0.128483, 0 to 153.992728, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.140538, 10 to 157.907342, 20 fc rgb "#FF0000"
set object 25 rect from 0.105850, 10 to 116.972182, 20 fc rgb "#00FF00"
set object 26 rect from 0.107843, 10 to 117.708955, 20 fc rgb "#0000FF"
set object 27 rect from 0.108280, 10 to 118.551300, 20 fc rgb "#FFFF00"
set object 28 rect from 0.109106, 10 to 119.900787, 20 fc rgb "#FF00FF"
set object 29 rect from 0.110322, 10 to 122.165529, 20 fc rgb "#808080"
set object 30 rect from 0.112408, 10 to 122.845713, 20 fc rgb "#800080"
set object 31 rect from 0.113265, 10 to 123.488898, 20 fc rgb "#008080"
set object 32 rect from 0.113594, 10 to 152.251458, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.138820, 20 to 156.664481, 30 fc rgb "#FF0000"
set object 34 rect from 0.091095, 20 to 100.681488, 30 fc rgb "#00FF00"
set object 35 rect from 0.092883, 20 to 101.334460, 30 fc rgb "#0000FF"
set object 36 rect from 0.093234, 20 to 103.366312, 30 fc rgb "#FFFF00"
set object 37 rect from 0.095106, 20 to 104.744089, 30 fc rgb "#FF00FF"
set object 38 rect from 0.096364, 20 to 106.377622, 30 fc rgb "#808080"
set object 39 rect from 0.097878, 20 to 107.083923, 30 fc rgb "#800080"
set object 40 rect from 0.098790, 20 to 107.772816, 30 fc rgb "#008080"
set object 41 rect from 0.099169, 20 to 150.579832, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.137379, 30 to 155.149575, 40 fc rgb "#FF0000"
set object 43 rect from 0.073958, 30 to 82.103193, 40 fc rgb "#00FF00"
set object 44 rect from 0.075815, 30 to 82.789905, 40 fc rgb "#0000FF"
set object 45 rect from 0.076192, 30 to 84.819575, 40 fc rgb "#FFFF00"
set object 46 rect from 0.078059, 30 to 86.210415, 40 fc rgb "#FF00FF"
set object 47 rect from 0.079334, 30 to 87.811304, 40 fc rgb "#808080"
set object 48 rect from 0.080816, 30 to 88.553517, 40 fc rgb "#800080"
set object 49 rect from 0.081775, 30 to 89.242410, 40 fc rgb "#008080"
set object 50 rect from 0.082143, 30 to 148.969156, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.135909, 40 to 154.050403, 50 fc rgb "#FF0000"
set object 52 rect from 0.058004, 40 to 65.257469, 50 fc rgb "#00FF00"
set object 53 rect from 0.060327, 40 to 65.936558, 50 fc rgb "#0000FF"
set object 54 rect from 0.060709, 40 to 68.116421, 50 fc rgb "#FFFF00"
set object 55 rect from 0.062714, 40 to 69.774977, 50 fc rgb "#FF00FF"
set object 56 rect from 0.064247, 40 to 71.474894, 50 fc rgb "#808080"
set object 57 rect from 0.065814, 40 to 72.211675, 50 fc rgb "#800080"
set object 58 rect from 0.066761, 40 to 72.973478, 50 fc rgb "#008080"
set object 59 rect from 0.067202, 40 to 147.251831, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.134207, 50 to 154.814396, 60 fc rgb "#FF0000"
set object 61 rect from 0.033827, 50 to 39.817469, 60 fc rgb "#00FF00"
set object 62 rect from 0.037182, 50 to 41.099477, 60 fc rgb "#0000FF"
set object 63 rect from 0.037903, 50 to 44.883480, 60 fc rgb "#FFFF00"
set object 64 rect from 0.041406, 50 to 46.954512, 60 fc rgb "#FF00FF"
set object 65 rect from 0.043281, 50 to 48.850321, 60 fc rgb "#808080"
set object 66 rect from 0.045050, 50 to 49.731837, 60 fc rgb "#800080"
set object 67 rect from 0.046256, 50 to 51.080237, 60 fc rgb "#008080"
set object 68 rect from 0.047078, 50 to 145.015379, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
