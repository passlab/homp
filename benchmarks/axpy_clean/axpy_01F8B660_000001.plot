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

set object 15 rect from 0.828182, 0 to 155.210643, 10 fc rgb "#FF0000"
set object 16 rect from 0.287897, 0 to 53.940153, 10 fc rgb "#00FF00"
set object 17 rect from 0.303525, 0 to 54.759976, 10 fc rgb "#0000FF"
set object 18 rect from 0.306464, 0 to 55.721302, 10 fc rgb "#FFFF00"
set object 19 rect from 0.312089, 0 to 57.205116, 10 fc rgb "#FF00FF"
set object 20 rect from 0.320346, 0 to 61.010301, 10 fc rgb "#808080"
set object 21 rect from 0.341554, 0 to 61.756327, 10 fc rgb "#800080"
set object 22 rect from 0.347134, 0 to 62.440022, 10 fc rgb "#008080"
set object 23 rect from 0.349431, 0 to 147.541882, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.838355, 10 to 156.480233, 20 fc rgb "#FF0000"
set object 25 rect from 0.395087, 10 to 72.593151, 20 fc rgb "#00FF00"
set object 26 rect from 0.407674, 10 to 73.328428, 20 fc rgb "#0000FF"
set object 27 rect from 0.410133, 10 to 74.187660, 20 fc rgb "#FFFF00"
set object 28 rect from 0.415045, 10 to 75.417841, 20 fc rgb "#FF00FF"
set object 29 rect from 0.421804, 10 to 79.166783, 20 fc rgb "#808080"
set object 30 rect from 0.442865, 10 to 79.784024, 20 fc rgb "#800080"
set object 31 rect from 0.447874, 10 to 80.401982, 20 fc rgb "#008080"
set object 32 rect from 0.449701, 10 to 149.484058, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.937329, 20 to 175.573859, 30 fc rgb "#FF0000"
set object 34 rect from 0.608365, 20 to 110.535193, 30 fc rgb "#00FF00"
set object 35 rect from 0.619588, 20 to 111.259367, 30 fc rgb "#0000FF"
set object 36 rect from 0.621904, 20 to 113.383707, 30 fc rgb "#FFFF00"
set object 37 rect from 0.633815, 20 to 114.853730, 30 fc rgb "#FF00FF"
set object 38 rect from 0.641966, 20 to 118.415133, 30 fc rgb "#808080"
set object 39 rect from 0.661875, 20 to 119.056019, 30 fc rgb "#800080"
set object 40 rect from 0.667065, 20 to 119.717860, 30 fc rgb "#008080"
set object 41 rect from 0.669163, 20 to 167.290184, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.918084, 30 to 172.659072, 40 fc rgb "#FF0000"
set object 43 rect from 0.500665, 30 to 91.618176, 40 fc rgb "#00FF00"
set object 44 rect from 0.513924, 30 to 92.464329, 40 fc rgb "#0000FF"
set object 45 rect from 0.517143, 30 to 94.656197, 40 fc rgb "#FFFF00"
set object 46 rect from 0.529303, 30 to 96.406718, 40 fc rgb "#FF00FF"
set object 47 rect from 0.539141, 30 to 100.058576, 40 fc rgb "#808080"
set object 48 rect from 0.559608, 30 to 100.790095, 40 fc rgb "#800080"
set object 49 rect from 0.565217, 30 to 101.590036, 40 fc rgb "#008080"
set object 50 rect from 0.567960, 30 to 163.601248, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.948073, 40 to 177.500631, 50 fc rgb "#FF0000"
set object 52 rect from 0.716186, 40 to 129.832837, 50 fc rgb "#00FF00"
set object 53 rect from 0.727101, 40 to 130.476047, 50 fc rgb "#0000FF"
set object 54 rect from 0.729204, 40 to 132.655199, 50 fc rgb "#FFFF00"
set object 55 rect from 0.741412, 40 to 134.124147, 50 fc rgb "#FF00FF"
set object 56 rect from 0.749554, 40 to 137.659399, 50 fc rgb "#808080"
set object 57 rect from 0.769288, 40 to 138.338973, 50 fc rgb "#800080"
set object 58 rect from 0.774713, 40 to 139.063149, 50 fc rgb "#008080"
set object 59 rect from 0.777141, 40 to 169.236303, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.816629, 50 to 156.487039, 60 fc rgb "#FF0000"
set object 61 rect from 0.144795, 50 to 29.147345, 60 fc rgb "#00FF00"
set object 62 rect from 0.165758, 50 to 30.283314, 60 fc rgb "#0000FF"
set object 63 rect from 0.169910, 50 to 33.359128, 60 fc rgb "#FFFF00"
set object 64 rect from 0.187267, 50 to 35.723129, 60 fc rgb "#FF00FF"
set object 65 rect from 0.200429, 50 to 39.556793, 60 fc rgb "#808080"
set object 66 rect from 0.221867, 50 to 40.450234, 60 fc rgb "#800080"
set object 67 rect from 0.229241, 50 to 41.731463, 60 fc rgb "#008080"
set object 68 rect from 0.233886, 50 to 145.194717, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
