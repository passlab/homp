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

set object 15 rect from 0.133590, 0 to 158.662756, 10 fc rgb "#FF0000"
set object 16 rect from 0.096595, 0 to 113.974850, 10 fc rgb "#00FF00"
set object 17 rect from 0.098998, 0 to 114.841037, 10 fc rgb "#0000FF"
set object 18 rect from 0.099419, 0 to 115.728042, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100186, 0 to 116.949263, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101255, 0 to 118.443406, 10 fc rgb "#808080"
set object 21 rect from 0.102561, 0 to 119.107214, 10 fc rgb "#800080"
set object 22 rect from 0.103371, 0 to 119.752519, 10 fc rgb "#008080"
set object 23 rect from 0.103670, 0 to 153.850724, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.128596, 10 to 153.589366, 20 fc rgb "#FF0000"
set object 25 rect from 0.049272, 10 to 59.647859, 20 fc rgb "#00FF00"
set object 26 rect from 0.051942, 10 to 60.674796, 20 fc rgb "#0000FF"
set object 27 rect from 0.052598, 10 to 61.736427, 20 fc rgb "#FFFF00"
set object 28 rect from 0.053537, 10 to 63.197035, 20 fc rgb "#FF00FF"
set object 29 rect from 0.054813, 10 to 64.739748, 20 fc rgb "#808080"
set object 30 rect from 0.056145, 10 to 65.527298, 20 fc rgb "#800080"
set object 31 rect from 0.057065, 10 to 66.239676, 20 fc rgb "#008080"
set object 32 rect from 0.057408, 10 to 147.914615, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.126632, 20 to 153.997597, 30 fc rgb "#FF0000"
set object 34 rect from 0.027411, 20 to 35.176022, 30 fc rgb "#00FF00"
set object 35 rect from 0.030945, 20 to 36.815883, 30 fc rgb "#0000FF"
set object 36 rect from 0.031967, 20 to 38.904447, 30 fc rgb "#FFFF00"
set object 37 rect from 0.033815, 20 to 40.950226, 30 fc rgb "#FF00FF"
set object 38 rect from 0.035539, 20 to 42.721922, 30 fc rgb "#808080"
set object 39 rect from 0.037088, 20 to 43.859875, 30 fc rgb "#800080"
set object 40 rect from 0.038523, 20 to 45.350554, 30 fc rgb "#008080"
set object 41 rect from 0.039350, 20 to 145.154148, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.131939, 30 to 156.980101, 40 fc rgb "#FF0000"
set object 43 rect from 0.081602, 30 to 96.488027, 40 fc rgb "#00FF00"
set object 44 rect from 0.083799, 30 to 97.248979, 40 fc rgb "#0000FF"
set object 45 rect from 0.084206, 30 to 98.296729, 40 fc rgb "#FFFF00"
set object 46 rect from 0.085115, 30 to 99.635908, 40 fc rgb "#FF00FF"
set object 47 rect from 0.086267, 30 to 100.985499, 40 fc rgb "#808080"
set object 48 rect from 0.087439, 30 to 101.704813, 40 fc rgb "#800080"
set object 49 rect from 0.088339, 30 to 102.426445, 40 fc rgb "#008080"
set object 50 rect from 0.088700, 30 to 151.940250, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.135166, 40 to 160.525809, 50 fc rgb "#FF0000"
set object 52 rect from 0.112725, 40 to 132.453913, 50 fc rgb "#00FF00"
set object 53 rect from 0.114899, 40 to 133.210234, 50 fc rgb "#0000FF"
set object 54 rect from 0.115300, 40 to 134.180506, 50 fc rgb "#FFFF00"
set object 55 rect from 0.116145, 40 to 135.442203, 50 fc rgb "#FF00FF"
set object 56 rect from 0.117234, 40 to 136.809139, 50 fc rgb "#808080"
set object 57 rect from 0.118416, 40 to 137.490291, 50 fc rgb "#800080"
set object 58 rect from 0.119295, 40 to 138.263961, 50 fc rgb "#008080"
set object 59 rect from 0.119672, 40 to 155.711464, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.130323, 50 to 155.483636, 60 fc rgb "#FF0000"
set object 61 rect from 0.065667, 50 to 78.138489, 60 fc rgb "#00FF00"
set object 62 rect from 0.068003, 50 to 79.105288, 60 fc rgb "#0000FF"
set object 63 rect from 0.068519, 50 to 80.212017, 60 fc rgb "#FFFF00"
set object 64 rect from 0.069487, 50 to 81.725822, 60 fc rgb "#FF00FF"
set object 65 rect from 0.070797, 50 to 83.141328, 60 fc rgb "#808080"
set object 66 rect from 0.072029, 50 to 83.868742, 60 fc rgb "#800080"
set object 67 rect from 0.072908, 50 to 84.663228, 60 fc rgb "#008080"
set object 68 rect from 0.073331, 50 to 149.943045, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
