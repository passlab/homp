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

set object 15 rect from 0.164253, 0 to 157.369112, 10 fc rgb "#FF0000"
set object 16 rect from 0.141468, 0 to 134.452917, 10 fc rgb "#00FF00"
set object 17 rect from 0.143775, 0 to 135.067897, 10 fc rgb "#0000FF"
set object 18 rect from 0.144189, 0 to 135.741939, 10 fc rgb "#FFFF00"
set object 19 rect from 0.144929, 0 to 136.717828, 10 fc rgb "#FF00FF"
set object 20 rect from 0.145952, 0 to 138.000256, 10 fc rgb "#808080"
set object 21 rect from 0.147331, 0 to 138.510244, 10 fc rgb "#800080"
set object 22 rect from 0.148116, 0 to 139.021153, 10 fc rgb "#008080"
set object 23 rect from 0.148415, 0 to 153.377385, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.162602, 10 to 156.077296, 20 fc rgb "#FF0000"
set object 25 rect from 0.124231, 10 to 118.383978, 20 fc rgb "#00FF00"
set object 26 rect from 0.126644, 10 to 119.086126, 20 fc rgb "#0000FF"
set object 27 rect from 0.127152, 10 to 119.850157, 20 fc rgb "#FFFF00"
set object 28 rect from 0.128016, 10 to 120.988228, 20 fc rgb "#FF00FF"
set object 29 rect from 0.129205, 10 to 122.187232, 20 fc rgb "#808080"
set object 30 rect from 0.130482, 10 to 122.788159, 20 fc rgb "#800080"
set object 31 rect from 0.131408, 10 to 123.371234, 20 fc rgb "#008080"
set object 32 rect from 0.131720, 10 to 151.874634, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.160852, 20 to 169.482868, 30 fc rgb "#FF0000"
set object 34 rect from 0.090581, 20 to 86.516039, 30 fc rgb "#00FF00"
set object 35 rect from 0.092631, 20 to 87.101964, 30 fc rgb "#0000FF"
set object 36 rect from 0.093026, 20 to 87.887563, 30 fc rgb "#FFFF00"
set object 37 rect from 0.093864, 20 to 104.069011, 30 fc rgb "#FF00FF"
set object 38 rect from 0.111124, 20 to 105.287684, 30 fc rgb "#808080"
set object 39 rect from 0.112447, 20 to 105.852040, 30 fc rgb "#800080"
set object 40 rect from 0.113315, 20 to 106.604812, 30 fc rgb "#008080"
set object 41 rect from 0.113836, 20 to 150.341878, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.159327, 30 to 152.902097, 40 fc rgb "#FF0000"
set object 43 rect from 0.074400, 30 to 71.418295, 40 fc rgb "#00FF00"
set object 44 rect from 0.076542, 30 to 72.009834, 40 fc rgb "#0000FF"
set object 45 rect from 0.076927, 30 to 72.801050, 40 fc rgb "#FFFF00"
set object 46 rect from 0.077772, 30 to 73.925990, 40 fc rgb "#FF00FF"
set object 47 rect from 0.078969, 30 to 74.990024, 40 fc rgb "#808080"
set object 48 rect from 0.080129, 30 to 75.634060, 40 fc rgb "#800080"
set object 49 rect from 0.081082, 30 to 76.247140, 40 fc rgb "#008080"
set object 50 rect from 0.081448, 30 to 148.823202, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.157623, 40 to 151.501517, 50 fc rgb "#FF0000"
set object 52 rect from 0.057790, 40 to 56.095562, 50 fc rgb "#00FF00"
set object 53 rect from 0.060204, 40 to 56.747085, 50 fc rgb "#0000FF"
set object 54 rect from 0.060646, 40 to 57.586131, 50 fc rgb "#FFFF00"
set object 55 rect from 0.061542, 40 to 58.802011, 50 fc rgb "#FF00FF"
set object 56 rect from 0.062856, 40 to 59.926950, 50 fc rgb "#808080"
set object 57 rect from 0.064065, 40 to 60.535365, 50 fc rgb "#800080"
set object 58 rect from 0.064956, 40 to 61.173757, 50 fc rgb "#008080"
set object 59 rect from 0.065400, 40 to 147.179837, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.155736, 50 to 152.730527, 60 fc rgb "#FF0000"
set object 61 rect from 0.033461, 50 to 34.145002, 60 fc rgb "#00FF00"
set object 62 rect from 0.037096, 50 to 35.513704, 60 fc rgb "#0000FF"
set object 63 rect from 0.038014, 50 to 37.754224, 60 fc rgb "#FFFF00"
set object 64 rect from 0.040434, 50 to 39.766935, 60 fc rgb "#FF00FF"
set object 65 rect from 0.042546, 50 to 41.064366, 60 fc rgb "#808080"
set object 66 rect from 0.043949, 50 to 41.904334, 60 fc rgb "#800080"
set object 67 rect from 0.045225, 50 to 43.338636, 60 fc rgb "#008080"
set object 68 rect from 0.046393, 50 to 145.109937, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
