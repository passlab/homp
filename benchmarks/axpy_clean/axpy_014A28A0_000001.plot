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

set object 15 rect from 0.129025, 0 to 162.264658, 10 fc rgb "#FF0000"
set object 16 rect from 0.106649, 0 to 132.039748, 10 fc rgb "#00FF00"
set object 17 rect from 0.108859, 0 to 132.741862, 10 fc rgb "#0000FF"
set object 18 rect from 0.109203, 0 to 133.638668, 10 fc rgb "#FFFF00"
set object 19 rect from 0.109939, 0 to 134.861584, 10 fc rgb "#FF00FF"
set object 20 rect from 0.110943, 0 to 137.500895, 10 fc rgb "#808080"
set object 21 rect from 0.113117, 0 to 138.143383, 10 fc rgb "#800080"
set object 22 rect from 0.113896, 0 to 138.772485, 10 fc rgb "#008080"
set object 23 rect from 0.114158, 0 to 156.333816, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.127471, 10 to 160.997939, 20 fc rgb "#FF0000"
set object 25 rect from 0.090934, 10 to 113.049856, 20 fc rgb "#00FF00"
set object 26 rect from 0.093252, 10 to 113.845665, 20 fc rgb "#0000FF"
set object 27 rect from 0.093671, 10 to 114.877538, 20 fc rgb "#FFFF00"
set object 28 rect from 0.094557, 10 to 116.334088, 20 fc rgb "#FF00FF"
set object 29 rect from 0.095755, 10 to 119.124286, 20 fc rgb "#808080"
set object 30 rect from 0.098044, 10 to 119.901841, 20 fc rgb "#800080"
set object 31 rect from 0.098910, 10 to 120.625856, 20 fc rgb "#008080"
set object 32 rect from 0.099252, 10 to 154.277369, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.125640, 20 to 159.918604, 30 fc rgb "#FF0000"
set object 34 rect from 0.075133, 20 to 93.651107, 30 fc rgb "#00FF00"
set object 35 rect from 0.077313, 20 to 94.359304, 30 fc rgb "#0000FF"
set object 36 rect from 0.077661, 20 to 96.710223, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079590, 20 to 98.266553, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080866, 20 to 100.769579, 30 fc rgb "#808080"
set object 39 rect from 0.082932, 20 to 101.530099, 30 fc rgb "#800080"
set object 40 rect from 0.083812, 20 to 102.250465, 30 fc rgb "#008080"
set object 41 rect from 0.084150, 20 to 152.212403, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.124041, 30 to 158.061718, 40 fc rgb "#FF0000"
set object 43 rect from 0.057409, 30 to 72.051106, 40 fc rgb "#00FF00"
set object 44 rect from 0.059586, 30 to 72.795807, 40 fc rgb "#0000FF"
set object 45 rect from 0.059938, 30 to 75.166199, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061890, 30 to 76.744430, 40 fc rgb "#FF00FF"
set object 47 rect from 0.063186, 30 to 79.265708, 40 fc rgb "#808080"
set object 48 rect from 0.065258, 30 to 80.056650, 40 fc rgb "#800080"
set object 49 rect from 0.066179, 30 to 80.832990, 40 fc rgb "#008080"
set object 50 rect from 0.066550, 30 to 150.242351, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.122450, 40 to 156.226734, 50 fc rgb "#FF0000"
set object 52 rect from 0.040454, 40 to 51.749467, 50 fc rgb "#00FF00"
set object 53 rect from 0.042871, 40 to 52.503904, 50 fc rgb "#0000FF"
set object 54 rect from 0.043262, 40 to 55.011796, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045335, 40 to 56.585161, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046639, 40 to 59.117389, 50 fc rgb "#808080"
set object 57 rect from 0.048740, 40 to 59.903464, 50 fc rgb "#800080"
set object 58 rect from 0.049625, 40 to 60.729693, 50 fc rgb "#008080"
set object 59 rect from 0.050066, 40 to 148.243096, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.120663, 50 to 158.168795, 60 fc rgb "#FF0000"
set object 61 rect from 0.016958, 50 to 24.251478, 60 fc rgb "#00FF00"
set object 62 rect from 0.020383, 50 to 25.653269, 60 fc rgb "#0000FF"
set object 63 rect from 0.021212, 50 to 30.555889, 60 fc rgb "#FFFF00"
set object 64 rect from 0.025280, 50 to 32.877606, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027154, 50 to 35.694574, 60 fc rgb "#808080"
set object 66 rect from 0.029488, 50 to 36.687510, 60 fc rgb "#800080"
set object 67 rect from 0.030708, 50 to 38.119723, 60 fc rgb "#008080"
set object 68 rect from 0.031464, 50 to 145.426126, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
