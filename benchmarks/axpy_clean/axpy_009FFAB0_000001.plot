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

set object 15 rect from 0.132346, 0 to 159.557807, 10 fc rgb "#FF0000"
set object 16 rect from 0.111253, 0 to 133.853488, 10 fc rgb "#00FF00"
set object 17 rect from 0.113583, 0 to 134.625383, 10 fc rgb "#0000FF"
set object 18 rect from 0.114008, 0 to 135.496477, 10 fc rgb "#FFFF00"
set object 19 rect from 0.114748, 0 to 136.683275, 10 fc rgb "#FF00FF"
set object 20 rect from 0.115751, 0 to 137.185613, 10 fc rgb "#808080"
set object 21 rect from 0.116190, 0 to 137.812090, 10 fc rgb "#800080"
set object 22 rect from 0.116961, 0 to 138.418488, 10 fc rgb "#008080"
set object 23 rect from 0.117238, 0 to 155.776467, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.130693, 10 to 158.092151, 20 fc rgb "#FF0000"
set object 25 rect from 0.095848, 10 to 115.517959, 20 fc rgb "#00FF00"
set object 26 rect from 0.098151, 10 to 116.477755, 20 fc rgb "#0000FF"
set object 27 rect from 0.098652, 10 to 117.413948, 20 fc rgb "#FFFF00"
set object 28 rect from 0.099478, 10 to 118.786253, 20 fc rgb "#FF00FF"
set object 29 rect from 0.100640, 10 to 119.364258, 20 fc rgb "#808080"
set object 30 rect from 0.101124, 10 to 120.132559, 20 fc rgb "#800080"
set object 31 rect from 0.102027, 10 to 120.831182, 20 fc rgb "#008080"
set object 32 rect from 0.102363, 10 to 153.817844, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129058, 20 to 155.860378, 30 fc rgb "#FF0000"
set object 34 rect from 0.082085, 20 to 99.113858, 30 fc rgb "#00FF00"
set object 35 rect from 0.084214, 20 to 99.883357, 30 fc rgb "#0000FF"
set object 36 rect from 0.084616, 20 to 100.892753, 30 fc rgb "#FFFF00"
set object 37 rect from 0.085467, 20 to 102.116188, 30 fc rgb "#FF00FF"
set object 38 rect from 0.086519, 20 to 102.662277, 30 fc rgb "#808080"
set object 39 rect from 0.086980, 20 to 103.291079, 30 fc rgb "#800080"
set object 40 rect from 0.087759, 20 to 103.974343, 30 fc rgb "#008080"
set object 41 rect from 0.088087, 20 to 151.949120, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127450, 30 to 154.242330, 40 fc rgb "#FF0000"
set object 43 rect from 0.065369, 30 to 79.460345, 40 fc rgb "#00FF00"
set object 44 rect from 0.067608, 30 to 80.245204, 40 fc rgb "#0000FF"
set object 45 rect from 0.068018, 30 to 81.311385, 40 fc rgb "#FFFF00"
set object 46 rect from 0.068915, 30 to 82.688480, 40 fc rgb "#FF00FF"
set object 47 rect from 0.070067, 30 to 83.229849, 40 fc rgb "#808080"
set object 48 rect from 0.070526, 30 to 83.900008, 40 fc rgb "#800080"
set object 49 rect from 0.071347, 30 to 84.573760, 40 fc rgb "#008080"
set object 50 rect from 0.071668, 30 to 149.949141, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.125667, 40 to 151.964550, 50 fc rgb "#FF0000"
set object 52 rect from 0.049769, 40 to 61.284395, 50 fc rgb "#00FF00"
set object 53 rect from 0.052212, 40 to 62.021979, 50 fc rgb "#0000FF"
set object 54 rect from 0.052585, 40 to 63.090485, 50 fc rgb "#FFFF00"
set object 55 rect from 0.053501, 40 to 64.406074, 50 fc rgb "#FF00FF"
set object 56 rect from 0.054615, 40 to 64.886007, 50 fc rgb "#808080"
set object 57 rect from 0.055022, 40 to 65.581036, 50 fc rgb "#800080"
set object 58 rect from 0.055881, 40 to 66.325736, 50 fc rgb "#008080"
set object 59 rect from 0.056241, 40 to 147.849821, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.123878, 50 to 153.226734, 60 fc rgb "#FF0000"
set object 61 rect from 0.027141, 50 to 35.392175, 60 fc rgb "#00FF00"
set object 62 rect from 0.030475, 50 to 37.067080, 60 fc rgb "#0000FF"
set object 63 rect from 0.031486, 50 to 39.559956, 60 fc rgb "#FFFF00"
set object 64 rect from 0.033659, 50 to 41.694714, 60 fc rgb "#FF00FF"
set object 65 rect from 0.035415, 50 to 42.487815, 60 fc rgb "#808080"
set object 66 rect from 0.036081, 50 to 43.368420, 60 fc rgb "#800080"
set object 67 rect from 0.037394, 50 to 44.836542, 60 fc rgb "#008080"
set object 68 rect from 0.038067, 50 to 145.278953, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
