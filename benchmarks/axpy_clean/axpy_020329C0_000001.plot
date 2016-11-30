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

set object 15 rect from 0.126948, 0 to 160.739913, 10 fc rgb "#FF0000"
set object 16 rect from 0.106053, 0 to 133.055424, 10 fc rgb "#00FF00"
set object 17 rect from 0.108261, 0 to 133.866926, 10 fc rgb "#0000FF"
set object 18 rect from 0.108655, 0 to 134.759879, 10 fc rgb "#FFFF00"
set object 19 rect from 0.109380, 0 to 136.017857, 10 fc rgb "#FF00FF"
set object 20 rect from 0.110397, 0 to 137.495380, 10 fc rgb "#808080"
set object 21 rect from 0.111597, 0 to 138.115750, 10 fc rgb "#800080"
set object 22 rect from 0.112352, 0 to 138.770670, 10 fc rgb "#008080"
set object 23 rect from 0.112649, 0 to 155.848592, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.125360, 10 to 159.326530, 20 fc rgb "#FF0000"
set object 25 rect from 0.090899, 10 to 114.602309, 20 fc rgb "#00FF00"
set object 26 rect from 0.093265, 10 to 115.490300, 20 fc rgb "#0000FF"
set object 27 rect from 0.093755, 10 to 116.414091, 20 fc rgb "#FFFF00"
set object 28 rect from 0.094536, 10 to 117.890364, 20 fc rgb "#FF00FF"
set object 29 rect from 0.095730, 10 to 119.512191, 20 fc rgb "#808080"
set object 30 rect from 0.097038, 10 to 120.267015, 20 fc rgb "#800080"
set object 31 rect from 0.097903, 10 to 121.002064, 20 fc rgb "#008080"
set object 32 rect from 0.098230, 10 to 153.821015, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.123627, 20 to 156.847624, 30 fc rgb "#FF0000"
set object 34 rect from 0.071356, 20 to 90.211977, 30 fc rgb "#00FF00"
set object 35 rect from 0.073532, 20 to 91.074055, 30 fc rgb "#0000FF"
set object 36 rect from 0.073955, 20 to 92.045924, 30 fc rgb "#FFFF00"
set object 37 rect from 0.074749, 20 to 93.453168, 30 fc rgb "#FF00FF"
set object 38 rect from 0.075886, 20 to 94.848063, 30 fc rgb "#808080"
set object 39 rect from 0.077023, 20 to 95.488171, 30 fc rgb "#800080"
set object 40 rect from 0.077804, 20 to 96.272547, 30 fc rgb "#008080"
set object 41 rect from 0.078176, 20 to 151.920429, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.122133, 30 to 155.673486, 40 fc rgb "#FF0000"
set object 43 rect from 0.055833, 30 to 71.132353, 40 fc rgb "#00FF00"
set object 44 rect from 0.058153, 30 to 72.507510, 40 fc rgb "#0000FF"
set object 45 rect from 0.058902, 30 to 73.629821, 40 fc rgb "#FFFF00"
set object 46 rect from 0.059820, 30 to 74.982814, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060911, 30 to 76.380172, 40 fc rgb "#808080"
set object 48 rect from 0.062045, 30 to 77.196635, 40 fc rgb "#800080"
set object 49 rect from 0.062996, 30 to 78.005674, 40 fc rgb "#008080"
set object 50 rect from 0.063384, 30 to 149.952029, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.120549, 40 to 153.507705, 50 fc rgb "#FF0000"
set object 52 rect from 0.040375, 40 to 52.583047, 50 fc rgb "#00FF00"
set object 53 rect from 0.042997, 40 to 53.424174, 50 fc rgb "#0000FF"
set object 54 rect from 0.043427, 40 to 54.550198, 50 fc rgb "#FFFF00"
set object 55 rect from 0.044349, 40 to 56.105423, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045618, 40 to 57.550894, 50 fc rgb "#808080"
set object 57 rect from 0.046792, 40 to 58.285979, 50 fc rgb "#800080"
set object 58 rect from 0.047697, 40 to 59.203559, 50 fc rgb "#008080"
set object 59 rect from 0.048131, 40 to 147.938014, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.118736, 50 to 154.052872, 60 fc rgb "#FF0000"
set object 61 rect from 0.020094, 50 to 28.228479, 60 fc rgb "#00FF00"
set object 62 rect from 0.023404, 50 to 30.016811, 60 fc rgb "#0000FF"
set object 63 rect from 0.024466, 50 to 32.226920, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026293, 50 to 34.242185, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027893, 50 to 36.021879, 60 fc rgb "#808080"
set object 66 rect from 0.029341, 50 to 36.987573, 60 fc rgb "#800080"
set object 67 rect from 0.030582, 50 to 38.546510, 60 fc rgb "#008080"
set object 68 rect from 0.031403, 50 to 145.285104, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
