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

set object 15 rect from 0.118667, 0 to 161.832056, 10 fc rgb "#FF0000"
set object 16 rect from 0.097785, 0 to 132.916508, 10 fc rgb "#00FF00"
set object 17 rect from 0.100001, 0 to 133.761080, 10 fc rgb "#0000FF"
set object 18 rect from 0.100369, 0 to 134.729734, 10 fc rgb "#FFFF00"
set object 19 rect from 0.101098, 0 to 136.177380, 10 fc rgb "#FF00FF"
set object 20 rect from 0.102177, 0 to 136.715076, 10 fc rgb "#808080"
set object 21 rect from 0.102601, 0 to 137.411546, 10 fc rgb "#800080"
set object 22 rect from 0.103365, 0 to 138.133369, 10 fc rgb "#008080"
set object 23 rect from 0.103651, 0 to 157.587850, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.117076, 10 to 160.142908, 20 fc rgb "#FF0000"
set object 25 rect from 0.083079, 10 to 113.570098, 20 fc rgb "#00FF00"
set object 26 rect from 0.085487, 10 to 114.488051, 20 fc rgb "#0000FF"
set object 27 rect from 0.085921, 10 to 115.539428, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086739, 10 to 117.129839, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087915, 10 to 117.746254, 20 fc rgb "#808080"
set object 30 rect from 0.088390, 10 to 118.538789, 20 fc rgb "#800080"
set object 31 rect from 0.089222, 10 to 119.355341, 20 fc rgb "#008080"
set object 32 rect from 0.089581, 10 to 155.351674, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.115381, 20 to 157.577179, 30 fc rgb "#FF0000"
set object 34 rect from 0.069943, 20 to 95.625969, 30 fc rgb "#00FF00"
set object 35 rect from 0.072037, 20 to 96.501226, 30 fc rgb "#0000FF"
set object 36 rect from 0.072443, 20 to 97.564614, 30 fc rgb "#FFFF00"
set object 37 rect from 0.073241, 20 to 99.009590, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074323, 20 to 99.524605, 30 fc rgb "#808080"
set object 39 rect from 0.074710, 20 to 100.215738, 30 fc rgb "#800080"
set object 40 rect from 0.075491, 20 to 101.111009, 30 fc rgb "#008080"
set object 41 rect from 0.075909, 20 to 153.159525, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.113706, 30 to 155.535798, 40 fc rgb "#FF0000"
set object 43 rect from 0.054432, 30 to 75.001363, 40 fc rgb "#00FF00"
set object 44 rect from 0.056632, 30 to 75.849934, 40 fc rgb "#0000FF"
set object 45 rect from 0.056963, 30 to 77.104116, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057904, 30 to 78.489052, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058944, 30 to 79.097463, 40 fc rgb "#808080"
set object 48 rect from 0.059408, 30 to 79.845970, 40 fc rgb "#800080"
set object 49 rect from 0.060230, 30 to 80.654517, 40 fc rgb "#008080"
set object 50 rect from 0.060575, 30 to 150.901998, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.112033, 40 to 153.501089, 50 fc rgb "#FF0000"
set object 52 rect from 0.040516, 40 to 56.877111, 50 fc rgb "#00FF00"
set object 53 rect from 0.043002, 40 to 57.807073, 50 fc rgb "#0000FF"
set object 54 rect from 0.043469, 40 to 59.063924, 50 fc rgb "#FFFF00"
set object 55 rect from 0.044383, 40 to 60.623645, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045578, 40 to 61.197365, 50 fc rgb "#808080"
set object 57 rect from 0.046012, 40 to 61.971221, 50 fc rgb "#800080"
set object 58 rect from 0.046929, 40 to 62.910522, 50 fc rgb "#008080"
set object 59 rect from 0.047293, 40 to 148.607113, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.110178, 50 to 154.986092, 60 fc rgb "#FF0000"
set object 61 rect from 0.019263, 50 to 29.957583, 60 fc rgb "#00FF00"
set object 62 rect from 0.022931, 50 to 31.638719, 60 fc rgb "#0000FF"
set object 63 rect from 0.023859, 50 to 35.121072, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026495, 50 to 37.353248, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028147, 50 to 38.273870, 60 fc rgb "#808080"
set object 66 rect from 0.028865, 50 to 39.270545, 60 fc rgb "#800080"
set object 67 rect from 0.030060, 50 to 40.804914, 60 fc rgb "#008080"
set object 68 rect from 0.030726, 50 to 145.621095, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
