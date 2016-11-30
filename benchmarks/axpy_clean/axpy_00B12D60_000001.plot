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

set object 15 rect from 0.210287, 0 to 155.009327, 10 fc rgb "#FF0000"
set object 16 rect from 0.074465, 0 to 53.367016, 10 fc rgb "#00FF00"
set object 17 rect from 0.076880, 0 to 53.865730, 10 fc rgb "#0000FF"
set object 18 rect from 0.077346, 0 to 54.395135, 10 fc rgb "#FFFF00"
set object 19 rect from 0.078158, 0 to 55.290726, 10 fc rgb "#FF00FF"
set object 20 rect from 0.079427, 0 to 61.826288, 10 fc rgb "#808080"
set object 21 rect from 0.088791, 0 to 62.292220, 10 fc rgb "#800080"
set object 22 rect from 0.089726, 0 to 62.723968, 10 fc rgb "#008080"
set object 23 rect from 0.090069, 0 to 146.156670, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.214020, 10 to 157.154127, 20 fc rgb "#FF0000"
set object 25 rect from 0.129762, 10 to 91.616452, 20 fc rgb "#00FF00"
set object 26 rect from 0.131716, 10 to 92.050995, 20 fc rgb "#0000FF"
set object 27 rect from 0.132092, 10 to 92.544127, 20 fc rgb "#FFFF00"
set object 28 rect from 0.132838, 10 to 93.333693, 20 fc rgb "#FF00FF"
set object 29 rect from 0.133932, 10 to 99.600021, 20 fc rgb "#808080"
set object 30 rect from 0.142948, 10 to 100.047120, 20 fc rgb "#800080"
set object 31 rect from 0.143820, 10 to 100.439819, 20 fc rgb "#008080"
set object 32 rect from 0.144123, 10 to 148.804373, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.215541, 20 to 163.056370, 30 fc rgb "#FF0000"
set object 34 rect from 0.151732, 20 to 106.858194, 30 fc rgb "#00FF00"
set object 35 rect from 0.153571, 20 to 107.303204, 30 fc rgb "#0000FF"
set object 36 rect from 0.153959, 20 to 109.752816, 30 fc rgb "#FFFF00"
set object 37 rect from 0.157495, 20 to 113.362368, 30 fc rgb "#FF00FF"
set object 38 rect from 0.162667, 20 to 119.609873, 30 fc rgb "#808080"
set object 39 rect from 0.171647, 20 to 120.130205, 30 fc rgb "#800080"
set object 40 rect from 0.172618, 20 to 120.640071, 30 fc rgb "#008080"
set object 41 rect from 0.173085, 20 to 149.967108, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.217062, 30 to 163.654819, 40 fc rgb "#FF0000"
set object 43 rect from 0.181086, 30 to 127.274683, 40 fc rgb "#00FF00"
set object 44 rect from 0.182848, 30 to 127.758742, 40 fc rgb "#0000FF"
set object 45 rect from 0.183287, 30 to 129.817068, 40 fc rgb "#FFFF00"
set object 46 rect from 0.186244, 30 to 133.358262, 40 fc rgb "#FF00FF"
set object 47 rect from 0.191336, 30 to 139.585531, 40 fc rgb "#808080"
set object 48 rect from 0.200247, 30 to 140.049374, 40 fc rgb "#800080"
set object 49 rect from 0.201189, 30 to 140.484614, 40 fc rgb "#008080"
set object 50 rect from 0.201551, 30 to 151.039160, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.208381, 40 to 160.839718, 50 fc rgb "#FF0000"
set object 52 rect from 0.036495, 40 to 27.936912, 50 fc rgb "#00FF00"
set object 53 rect from 0.040537, 40 to 28.942009, 50 fc rgb "#0000FF"
set object 54 rect from 0.041633, 40 to 32.330459, 50 fc rgb "#FFFF00"
set object 55 rect from 0.046515, 40 to 36.986266, 50 fc rgb "#FF00FF"
set object 56 rect from 0.053155, 40 to 43.475099, 50 fc rgb "#808080"
set object 57 rect from 0.062481, 40 to 44.080523, 50 fc rgb "#800080"
set object 58 rect from 0.063770, 40 to 44.970532, 50 fc rgb "#008080"
set object 59 rect from 0.064613, 40 to 144.760973, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.212185, 50 to 160.942251, 60 fc rgb "#FF0000"
set object 61 rect from 0.098918, 50 to 70.235328, 60 fc rgb "#00FF00"
set object 62 rect from 0.101084, 50 to 70.725674, 60 fc rgb "#0000FF"
set object 63 rect from 0.101538, 50 to 73.201093, 60 fc rgb "#FFFF00"
set object 64 rect from 0.105098, 50 to 76.962006, 60 fc rgb "#FF00FF"
set object 65 rect from 0.110486, 50 to 83.247167, 60 fc rgb "#808080"
set object 66 rect from 0.119522, 50 to 83.787725, 60 fc rgb "#800080"
set object 67 rect from 0.120512, 50 to 84.310853, 60 fc rgb "#008080"
set object 68 rect from 0.121022, 50 to 147.451227, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
