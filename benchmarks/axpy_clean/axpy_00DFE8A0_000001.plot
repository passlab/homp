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

set object 15 rect from 0.113060, 0 to 162.237271, 10 fc rgb "#FF0000"
set object 16 rect from 0.093892, 0 to 132.605199, 10 fc rgb "#00FF00"
set object 17 rect from 0.095710, 0 to 133.465885, 10 fc rgb "#0000FF"
set object 18 rect from 0.096097, 0 to 134.325203, 10 fc rgb "#FFFF00"
set object 19 rect from 0.096716, 0 to 135.629450, 10 fc rgb "#FF00FF"
set object 20 rect from 0.097654, 0 to 137.907014, 10 fc rgb "#808080"
set object 21 rect from 0.099294, 0 to 138.578614, 10 fc rgb "#800080"
set object 22 rect from 0.100030, 0 to 139.300251, 10 fc rgb "#008080"
set object 23 rect from 0.100298, 0 to 156.411234, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.111601, 10 to 160.529782, 20 fc rgb "#FF0000"
set object 25 rect from 0.079514, 10 to 112.652131, 20 fc rgb "#00FF00"
set object 26 rect from 0.081353, 10 to 113.483622, 20 fc rgb "#0000FF"
set object 27 rect from 0.081727, 10 to 114.422172, 20 fc rgb "#FFFF00"
set object 28 rect from 0.082440, 10 to 115.922487, 20 fc rgb "#FF00FF"
set object 29 rect from 0.083505, 10 to 118.273750, 20 fc rgb "#808080"
set object 30 rect from 0.085189, 10 to 119.042669, 20 fc rgb "#800080"
set object 31 rect from 0.085976, 10 to 119.789335, 20 fc rgb "#008080"
set object 32 rect from 0.086283, 10 to 154.306077, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.109991, 20 to 159.664891, 30 fc rgb "#FF0000"
set object 34 rect from 0.065501, 20 to 92.843663, 30 fc rgb "#00FF00"
set object 35 rect from 0.067114, 20 to 93.669581, 30 fc rgb "#0000FF"
set object 36 rect from 0.067477, 20 to 96.102893, 30 fc rgb "#FFFF00"
set object 37 rect from 0.069228, 20 to 97.557315, 30 fc rgb "#FF00FF"
set object 38 rect from 0.070272, 20 to 99.719471, 30 fc rgb "#808080"
set object 39 rect from 0.071827, 20 to 100.510643, 30 fc rgb "#800080"
set object 40 rect from 0.072634, 20 to 101.308776, 30 fc rgb "#008080"
set object 41 rect from 0.072975, 20 to 152.323289, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.108638, 30 to 157.709929, 40 fc rgb "#FF0000"
set object 43 rect from 0.050397, 30 to 72.348304, 40 fc rgb "#00FF00"
set object 44 rect from 0.052373, 30 to 73.150602, 40 fc rgb "#0000FF"
set object 45 rect from 0.052722, 30 to 75.578340, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054467, 30 to 77.125917, 40 fc rgb "#FF00FF"
set object 47 rect from 0.055581, 30 to 79.128181, 40 fc rgb "#808080"
set object 48 rect from 0.057038, 30 to 79.973576, 40 fc rgb "#800080"
set object 49 rect from 0.057909, 30 to 80.834282, 40 fc rgb "#008080"
set object 50 rect from 0.058266, 30 to 150.261228, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.107073, 40 to 155.834177, 50 fc rgb "#FF0000"
set object 52 rect from 0.035392, 40 to 51.793170, 50 fc rgb "#00FF00"
set object 53 rect from 0.037597, 40 to 52.649691, 50 fc rgb "#0000FF"
set object 54 rect from 0.038005, 40 to 55.237321, 50 fc rgb "#FFFF00"
set object 55 rect from 0.039841, 40 to 56.837753, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041005, 40 to 58.890054, 50 fc rgb "#808080"
set object 57 rect from 0.042473, 40 to 59.754925, 50 fc rgb "#800080"
set object 58 rect from 0.043359, 40 to 60.680981, 50 fc rgb "#008080"
set object 59 rect from 0.043789, 40 to 148.096295, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.105493, 50 to 156.651766, 60 fc rgb "#FF0000"
set object 61 rect from 0.015502, 50 to 24.978175, 60 fc rgb "#00FF00"
set object 62 rect from 0.018509, 50 to 26.573033, 60 fc rgb "#0000FF"
set object 63 rect from 0.019225, 50 to 30.709649, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022255, 50 to 33.016427, 60 fc rgb "#FF00FF"
set object 65 rect from 0.023885, 50 to 35.257835, 60 fc rgb "#808080"
set object 66 rect from 0.025489, 50 to 36.303462, 60 fc rgb "#800080"
set object 67 rect from 0.026630, 50 to 37.881641, 60 fc rgb "#008080"
set object 68 rect from 0.027388, 50 to 145.236122, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0