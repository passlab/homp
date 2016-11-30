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

set object 15 rect from 0.159521, 0 to 153.160105, 10 fc rgb "#FF0000"
set object 16 rect from 0.065590, 0 to 63.577548, 10 fc rgb "#00FF00"
set object 17 rect from 0.068190, 0 to 64.313790, 10 fc rgb "#0000FF"
set object 18 rect from 0.068697, 0 to 65.069681, 10 fc rgb "#FFFF00"
set object 19 rect from 0.069496, 0 to 66.104154, 10 fc rgb "#FF00FF"
set object 20 rect from 0.070613, 0 to 67.370280, 10 fc rgb "#808080"
set object 21 rect from 0.071961, 0 to 67.860753, 10 fc rgb "#800080"
set object 22 rect from 0.072723, 0 to 68.350333, 10 fc rgb "#008080"
set object 23 rect from 0.073008, 0 to 148.357302, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.156580, 10 to 154.093286, 20 fc rgb "#FF0000"
set object 25 rect from 0.039352, 10 to 39.841246, 20 fc rgb "#00FF00"
set object 26 rect from 0.043000, 10 to 41.512513, 20 fc rgb "#0000FF"
set object 27 rect from 0.044402, 10 to 44.462093, 20 fc rgb "#FFFF00"
set object 28 rect from 0.047564, 10 to 45.801365, 20 fc rgb "#FF00FF"
set object 29 rect from 0.048967, 10 to 47.325389, 20 fc rgb "#808080"
set object 30 rect from 0.050609, 10 to 47.987506, 20 fc rgb "#800080"
set object 31 rect from 0.051834, 10 to 48.813776, 20 fc rgb "#008080"
set object 32 rect from 0.052181, 10 to 145.581239, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.167538, 20 to 160.742689, 30 fc rgb "#FF0000"
set object 34 rect from 0.140644, 20 to 133.609462, 30 fc rgb "#00FF00"
set object 35 rect from 0.142910, 20 to 134.355989, 30 fc rgb "#0000FF"
set object 36 rect from 0.143403, 20 to 135.178486, 30 fc rgb "#FFFF00"
set object 37 rect from 0.144250, 20 to 136.411798, 30 fc rgb "#FF00FF"
set object 38 rect from 0.145561, 20 to 137.485625, 30 fc rgb "#808080"
set object 39 rect from 0.146717, 20 to 138.001450, 30 fc rgb "#800080"
set object 40 rect from 0.147577, 20 to 138.634527, 30 fc rgb "#008080"
set object 41 rect from 0.147937, 20 to 156.653627, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.164079, 30 to 157.594327, 40 fc rgb "#FF0000"
set object 43 rect from 0.105409, 30 to 100.659613, 40 fc rgb "#00FF00"
set object 44 rect from 0.107690, 30 to 101.342386, 40 fc rgb "#0000FF"
set object 45 rect from 0.108170, 30 to 102.156442, 40 fc rgb "#FFFF00"
set object 46 rect from 0.109037, 30 to 103.380335, 40 fc rgb "#FF00FF"
set object 47 rect from 0.110351, 30 to 104.465454, 40 fc rgb "#808080"
set object 48 rect from 0.111516, 30 to 105.046934, 40 fc rgb "#800080"
set object 49 rect from 0.112376, 30 to 105.650914, 40 fc rgb "#008080"
set object 50 rect from 0.112769, 30 to 153.272634, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.162216, 40 to 156.835613, 50 fc rgb "#FF0000"
set object 52 rect from 0.084560, 40 to 81.243104, 50 fc rgb "#00FF00"
set object 53 rect from 0.086991, 40 to 82.311313, 50 fc rgb "#0000FF"
set object 54 rect from 0.087881, 40 to 83.308276, 50 fc rgb "#FFFF00"
set object 55 rect from 0.088975, 40 to 84.856673, 50 fc rgb "#FF00FF"
set object 56 rect from 0.090610, 40 to 85.996184, 50 fc rgb "#808080"
set object 57 rect from 0.091840, 40 to 86.683624, 50 fc rgb "#800080"
set object 58 rect from 0.092809, 40 to 87.760274, 50 fc rgb "#008080"
set object 59 rect from 0.093720, 40 to 151.274990, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.165892, 50 to 159.178360, 60 fc rgb "#FF0000"
set object 61 rect from 0.122038, 50 to 116.015207, 60 fc rgb "#00FF00"
set object 62 rect from 0.124098, 50 to 116.669834, 60 fc rgb "#0000FF"
set object 63 rect from 0.124513, 50 to 117.468880, 60 fc rgb "#FFFF00"
set object 64 rect from 0.125362, 50 to 118.615882, 60 fc rgb "#FF00FF"
set object 65 rect from 0.126590, 50 to 119.692559, 60 fc rgb "#808080"
set object 66 rect from 0.127737, 50 to 120.300285, 60 fc rgb "#800080"
set object 67 rect from 0.128638, 50 to 120.876120, 60 fc rgb "#008080"
set object 68 rect from 0.129016, 50 to 154.891382, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
