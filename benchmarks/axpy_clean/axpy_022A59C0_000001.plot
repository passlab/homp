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

set object 15 rect from 0.118851, 0 to 159.419169, 10 fc rgb "#FF0000"
set object 16 rect from 0.101187, 0 to 134.443018, 10 fc rgb "#00FF00"
set object 17 rect from 0.102769, 0 to 135.116082, 10 fc rgb "#0000FF"
set object 18 rect from 0.103080, 0 to 135.848230, 10 fc rgb "#FFFF00"
set object 19 rect from 0.103643, 0 to 137.010696, 10 fc rgb "#FF00FF"
set object 20 rect from 0.104526, 0 to 138.136445, 10 fc rgb "#808080"
set object 21 rect from 0.105386, 0 to 138.720279, 10 fc rgb "#800080"
set object 22 rect from 0.106047, 0 to 139.293672, 10 fc rgb "#008080"
set object 23 rect from 0.106267, 0 to 155.395133, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.117646, 10 to 158.423277, 20 fc rgb "#FF0000"
set object 25 rect from 0.087832, 10 to 117.127905, 20 fc rgb "#00FF00"
set object 26 rect from 0.089579, 10 to 117.925627, 20 fc rgb "#0000FF"
set object 27 rect from 0.089981, 10 to 118.761395, 20 fc rgb "#FFFF00"
set object 28 rect from 0.090664, 10 to 120.178416, 20 fc rgb "#FF00FF"
set object 29 rect from 0.091708, 10 to 121.403876, 20 fc rgb "#808080"
set object 30 rect from 0.092657, 10 to 122.107127, 20 fc rgb "#800080"
set object 31 rect from 0.093406, 10 to 122.785470, 20 fc rgb "#008080"
set object 32 rect from 0.093695, 10 to 153.634366, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.114955, 20 to 155.531638, 30 fc rgb "#FF0000"
set object 34 rect from 0.058842, 20 to 79.636112, 30 fc rgb "#00FF00"
set object 35 rect from 0.061048, 20 to 80.571629, 30 fc rgb "#0000FF"
set object 36 rect from 0.061514, 20 to 81.646194, 30 fc rgb "#FFFF00"
set object 37 rect from 0.062346, 20 to 83.218020, 30 fc rgb "#FF00FF"
set object 38 rect from 0.063559, 20 to 84.607474, 30 fc rgb "#808080"
set object 39 rect from 0.064628, 20 to 85.330433, 30 fc rgb "#800080"
set object 40 rect from 0.065439, 20 to 86.157012, 30 fc rgb "#008080"
set object 41 rect from 0.065772, 20 to 150.062976, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.113490, 30 to 154.386142, 40 fc rgb "#FF0000"
set object 43 rect from 0.042493, 30 to 58.403243, 40 fc rgb "#00FF00"
set object 44 rect from 0.044852, 30 to 59.518474, 40 fc rgb "#0000FF"
set object 45 rect from 0.045486, 30 to 60.749173, 40 fc rgb "#FFFF00"
set object 46 rect from 0.046412, 30 to 62.429899, 40 fc rgb "#FF00FF"
set object 47 rect from 0.047714, 30 to 63.845629, 40 fc rgb "#808080"
set object 48 rect from 0.048802, 30 to 64.820484, 40 fc rgb "#800080"
set object 49 rect from 0.049822, 30 to 65.761202, 40 fc rgb "#008080"
set object 50 rect from 0.050272, 30 to 148.205079, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.111882, 40 to 155.010720, 50 fc rgb "#FF0000"
set object 52 rect from 0.021400, 40 to 32.244918, 50 fc rgb "#00FF00"
set object 53 rect from 0.025038, 40 to 33.967639, 50 fc rgb "#0000FF"
set object 54 rect from 0.026019, 40 to 36.623219, 50 fc rgb "#FFFF00"
set object 55 rect from 0.028058, 40 to 39.045281, 50 fc rgb "#FF00FF"
set object 56 rect from 0.029902, 40 to 40.647294, 50 fc rgb "#808080"
set object 57 rect from 0.031107, 40 to 41.643146, 50 fc rgb "#800080"
set object 58 rect from 0.032323, 40 to 43.346160, 50 fc rgb "#008080"
set object 59 rect from 0.033177, 40 to 145.616442, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116204, 50 to 157.276569, 60 fc rgb "#FF0000"
set object 61 rect from 0.074580, 50 to 100.474088, 60 fc rgb "#00FF00"
set object 62 rect from 0.076957, 50 to 101.395138, 60 fc rgb "#0000FF"
set object 63 rect from 0.077396, 50 to 102.625837, 60 fc rgb "#FFFF00"
set object 64 rect from 0.078324, 50 to 104.113711, 60 fc rgb "#FF00FF"
set object 65 rect from 0.079500, 50 to 105.625164, 60 fc rgb "#808080"
set object 66 rect from 0.080617, 50 to 106.287749, 60 fc rgb "#800080"
set object 67 rect from 0.081363, 50 to 107.043514, 60 fc rgb "#008080"
set object 68 rect from 0.081702, 50 to 151.927364, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
