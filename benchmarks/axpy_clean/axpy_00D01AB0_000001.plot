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

set object 15 rect from 0.117688, 0 to 155.167419, 10 fc rgb "#FF0000"
set object 16 rect from 0.078191, 0 to 102.996089, 10 fc rgb "#00FF00"
set object 17 rect from 0.079927, 0 to 103.758024, 10 fc rgb "#0000FF"
set object 18 rect from 0.080311, 0 to 104.460436, 10 fc rgb "#FFFF00"
set object 19 rect from 0.080852, 0 to 105.625930, 10 fc rgb "#FF00FF"
set object 20 rect from 0.081765, 0 to 106.117502, 10 fc rgb "#808080"
set object 21 rect from 0.082146, 0 to 106.738416, 10 fc rgb "#800080"
set object 22 rect from 0.082846, 0 to 107.369699, 10 fc rgb "#008080"
set object 23 rect from 0.083118, 0 to 151.615306, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.114877, 10 to 152.048549, 20 fc rgb "#FF0000"
set object 25 rect from 0.051614, 10 to 68.972383, 20 fc rgb "#00FF00"
set object 26 rect from 0.053632, 10 to 69.899859, 20 fc rgb "#0000FF"
set object 27 rect from 0.054154, 10 to 70.726445, 20 fc rgb "#FFFF00"
set object 28 rect from 0.054809, 10 to 72.126103, 20 fc rgb "#FF00FF"
set object 29 rect from 0.055882, 10 to 72.647437, 20 fc rgb "#808080"
set object 30 rect from 0.056286, 10 to 73.342061, 20 fc rgb "#800080"
set object 31 rect from 0.057012, 10 to 73.990153, 20 fc rgb "#008080"
set object 32 rect from 0.057335, 10 to 147.765612, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.113198, 20 to 153.086052, 30 fc rgb "#FF0000"
set object 34 rect from 0.033367, 20 to 46.536540, 30 fc rgb "#00FF00"
set object 35 rect from 0.036357, 20 to 47.914184, 30 fc rgb "#0000FF"
set object 36 rect from 0.037153, 20 to 50.408208, 30 fc rgb "#FFFF00"
set object 37 rect from 0.039114, 20 to 52.532251, 30 fc rgb "#FF00FF"
set object 38 rect from 0.040714, 20 to 53.313578, 30 fc rgb "#808080"
set object 39 rect from 0.041331, 20 to 54.163449, 30 fc rgb "#800080"
set object 40 rect from 0.042376, 20 to 55.326399, 30 fc rgb "#008080"
set object 41 rect from 0.042887, 20 to 145.334967, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.118924, 30 to 156.969401, 40 fc rgb "#FF0000"
set object 43 rect from 0.090328, 30 to 118.468612, 40 fc rgb "#00FF00"
set object 44 rect from 0.091886, 30 to 119.226653, 40 fc rgb "#0000FF"
set object 45 rect from 0.092267, 30 to 120.133427, 40 fc rgb "#FFFF00"
set object 46 rect from 0.092969, 30 to 121.340365, 40 fc rgb "#FF00FF"
set object 47 rect from 0.093912, 30 to 121.796971, 40 fc rgb "#808080"
set object 48 rect from 0.094266, 30 to 122.406280, 40 fc rgb "#800080"
set object 49 rect from 0.094944, 30 to 123.095738, 40 fc rgb "#008080"
set object 50 rect from 0.095279, 30 to 153.307300, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.120138, 40 to 158.349705, 50 fc rgb "#FF0000"
set object 52 rect from 0.102772, 40 to 134.400324, 50 fc rgb "#00FF00"
set object 53 rect from 0.104207, 40 to 135.171319, 50 fc rgb "#0000FF"
set object 54 rect from 0.104594, 40 to 136.018608, 50 fc rgb "#FFFF00"
set object 55 rect from 0.105249, 40 to 137.156962, 50 fc rgb "#FF00FF"
set object 56 rect from 0.106129, 40 to 137.560559, 50 fc rgb "#808080"
set object 57 rect from 0.106441, 40 to 138.128425, 50 fc rgb "#800080"
set object 58 rect from 0.107097, 40 to 138.785577, 50 fc rgb "#008080"
set object 59 rect from 0.107392, 40 to 154.873809, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116257, 50 to 153.928136, 60 fc rgb "#FF0000"
set object 61 rect from 0.065394, 50 to 86.448629, 60 fc rgb "#00FF00"
set object 62 rect from 0.067132, 50 to 87.391642, 60 fc rgb "#0000FF"
set object 63 rect from 0.067658, 50 to 88.395450, 60 fc rgb "#FFFF00"
set object 64 rect from 0.068443, 50 to 89.687741, 60 fc rgb "#FF00FF"
set object 65 rect from 0.069444, 50 to 90.175420, 60 fc rgb "#808080"
set object 66 rect from 0.069811, 50 to 90.796333, 60 fc rgb "#800080"
set object 67 rect from 0.070550, 50 to 91.536294, 60 fc rgb "#008080"
set object 68 rect from 0.070866, 50 to 149.682672, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
