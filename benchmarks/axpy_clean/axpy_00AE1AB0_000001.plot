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

set object 15 rect from 0.120327, 0 to 161.501093, 10 fc rgb "#FF0000"
set object 16 rect from 0.099427, 0 to 132.964046, 10 fc rgb "#00FF00"
set object 17 rect from 0.101711, 0 to 133.832542, 10 fc rgb "#0000FF"
set object 18 rect from 0.102129, 0 to 134.828304, 10 fc rgb "#FFFF00"
set object 19 rect from 0.102886, 0 to 136.301576, 10 fc rgb "#FF00FF"
set object 20 rect from 0.104010, 0 to 136.857831, 10 fc rgb "#808080"
set object 21 rect from 0.104447, 0 to 137.551869, 10 fc rgb "#800080"
set object 22 rect from 0.105240, 0 to 138.270774, 10 fc rgb "#008080"
set object 23 rect from 0.105516, 0 to 157.081238, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.118690, 10 to 159.699862, 20 fc rgb "#FF0000"
set object 25 rect from 0.084149, 10 to 112.972846, 20 fc rgb "#00FF00"
set object 26 rect from 0.086485, 10 to 113.934513, 20 fc rgb "#0000FF"
set object 27 rect from 0.086961, 10 to 114.977467, 20 fc rgb "#FFFF00"
set object 28 rect from 0.087796, 10 to 116.600330, 20 fc rgb "#FF00FF"
set object 29 rect from 0.089027, 10 to 117.252376, 20 fc rgb "#808080"
set object 30 rect from 0.089519, 10 to 118.042167, 20 fc rgb "#800080"
set object 31 rect from 0.090341, 10 to 118.797824, 20 fc rgb "#008080"
set object 32 rect from 0.090671, 10 to 154.860128, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.116939, 20 to 157.019540, 30 fc rgb "#FF0000"
set object 34 rect from 0.070701, 20 to 94.946933, 30 fc rgb "#00FF00"
set object 35 rect from 0.072735, 20 to 95.748531, 30 fc rgb "#0000FF"
set object 36 rect from 0.073117, 20 to 96.875468, 30 fc rgb "#FFFF00"
set object 37 rect from 0.073953, 20 to 98.318596, 30 fc rgb "#FF00FF"
set object 38 rect from 0.075053, 20 to 98.838137, 30 fc rgb "#808080"
set object 39 rect from 0.075449, 20 to 99.494093, 30 fc rgb "#800080"
set object 40 rect from 0.076202, 20 to 100.236653, 30 fc rgb "#008080"
set object 41 rect from 0.076539, 20 to 152.775472, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.115371, 30 to 155.202553, 40 fc rgb "#FF0000"
set object 43 rect from 0.054981, 30 to 74.530692, 40 fc rgb "#00FF00"
set object 44 rect from 0.057161, 30 to 75.300816, 40 fc rgb "#0000FF"
set object 45 rect from 0.057511, 30 to 76.553690, 40 fc rgb "#FFFF00"
set object 46 rect from 0.058476, 30 to 78.033569, 40 fc rgb "#FF00FF"
set object 47 rect from 0.059592, 30 to 78.576687, 40 fc rgb "#808080"
set object 48 rect from 0.060009, 30 to 79.264157, 40 fc rgb "#800080"
set object 49 rect from 0.060785, 30 to 80.034242, 40 fc rgb "#008080"
set object 50 rect from 0.061123, 30 to 150.633106, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.113733, 40 to 153.016827, 50 fc rgb "#FF0000"
set object 52 rect from 0.040379, 40 to 55.775357, 50 fc rgb "#00FF00"
set object 53 rect from 0.042861, 40 to 56.561199, 50 fc rgb "#0000FF"
set object 54 rect from 0.043224, 40 to 57.671089, 50 fc rgb "#FFFF00"
set object 55 rect from 0.044087, 40 to 59.255909, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045296, 40 to 59.863344, 50 fc rgb "#808080"
set object 57 rect from 0.045754, 40 to 60.537677, 50 fc rgb "#800080"
set object 58 rect from 0.046531, 40 to 61.320899, 50 fc rgb "#008080"
set object 59 rect from 0.046891, 40 to 148.401518, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.111876, 50 to 154.313061, 60 fc rgb "#FF0000"
set object 61 rect from 0.020606, 50 to 30.671592, 60 fc rgb "#00FF00"
set object 62 rect from 0.023905, 50 to 32.283938, 60 fc rgb "#0000FF"
set object 63 rect from 0.024751, 50 to 35.410296, 60 fc rgb "#FFFF00"
set object 64 rect from 0.027167, 50 to 37.578897, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028772, 50 to 38.530047, 60 fc rgb "#808080"
set object 66 rect from 0.029503, 50 to 39.498244, 60 fc rgb "#800080"
set object 67 rect from 0.030684, 50 to 40.915136, 60 fc rgb "#008080"
set object 68 rect from 0.031312, 50 to 145.555926, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
