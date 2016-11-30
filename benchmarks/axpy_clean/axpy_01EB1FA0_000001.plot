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

set object 15 rect from 0.178196, 0 to 158.579621, 10 fc rgb "#FF0000"
set object 16 rect from 0.154967, 0 to 133.077825, 10 fc rgb "#00FF00"
set object 17 rect from 0.156604, 0 to 133.535015, 10 fc rgb "#0000FF"
set object 18 rect from 0.156944, 0 to 133.987943, 10 fc rgb "#FFFF00"
set object 19 rect from 0.157486, 0 to 134.729484, 10 fc rgb "#FF00FF"
set object 20 rect from 0.158355, 0 to 140.070103, 10 fc rgb "#808080"
set object 21 rect from 0.164625, 0 to 140.460867, 10 fc rgb "#800080"
set object 22 rect from 0.165291, 0 to 140.837169, 10 fc rgb "#008080"
set object 23 rect from 0.165524, 0 to 151.314079, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.172681, 10 to 155.349552, 20 fc rgb "#FF0000"
set object 25 rect from 0.065260, 10 to 56.930826, 20 fc rgb "#00FF00"
set object 26 rect from 0.067166, 10 to 58.389222, 20 fc rgb "#0000FF"
set object 27 rect from 0.068697, 10 to 58.975813, 20 fc rgb "#FFFF00"
set object 28 rect from 0.069403, 10 to 59.872306, 20 fc rgb "#FF00FF"
set object 29 rect from 0.070445, 10 to 65.408725, 20 fc rgb "#808080"
set object 30 rect from 0.076950, 10 to 65.829302, 20 fc rgb "#800080"
set object 31 rect from 0.077669, 10 to 66.293292, 20 fc rgb "#008080"
set object 32 rect from 0.077988, 10 to 146.513240, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.175625, 20 to 160.561607, 30 fc rgb "#FF0000"
set object 34 rect from 0.108883, 20 to 93.798362, 30 fc rgb "#00FF00"
set object 35 rect from 0.110467, 20 to 94.264078, 30 fc rgb "#0000FF"
set object 36 rect from 0.110817, 20 to 96.598516, 30 fc rgb "#FFFF00"
set object 37 rect from 0.113586, 20 to 99.680459, 30 fc rgb "#FF00FF"
set object 38 rect from 0.117181, 20 to 104.877188, 30 fc rgb "#808080"
set object 39 rect from 0.123289, 20 to 105.364166, 30 fc rgb "#800080"
set object 40 rect from 0.124102, 20 to 105.888619, 30 fc rgb "#008080"
set object 41 rect from 0.124479, 20 to 149.024754, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.176945, 30 to 161.040948, 40 fc rgb "#FF0000"
set object 43 rect from 0.132318, 30 to 113.658189, 40 fc rgb "#00FF00"
set object 44 rect from 0.133809, 30 to 114.073641, 40 fc rgb "#0000FF"
set object 45 rect from 0.134091, 30 to 116.433630, 40 fc rgb "#FFFF00"
set object 46 rect from 0.136863, 30 to 118.925581, 40 fc rgb "#FF00FF"
set object 47 rect from 0.139800, 30 to 124.089961, 40 fc rgb "#808080"
set object 48 rect from 0.145853, 30 to 124.588027, 40 fc rgb "#800080"
set object 49 rect from 0.146677, 30 to 125.080104, 40 fc rgb "#008080"
set object 50 rect from 0.147022, 30 to 150.237111, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.174162, 40 to 159.339938, 50 fc rgb "#FF0000"
set object 52 rect from 0.085514, 40 to 74.022012, 50 fc rgb "#00FF00"
set object 53 rect from 0.087238, 40 to 74.499627, 50 fc rgb "#0000FF"
set object 54 rect from 0.087608, 40 to 76.847690, 50 fc rgb "#FFFF00"
set object 55 rect from 0.090396, 40 to 79.899820, 50 fc rgb "#FF00FF"
set object 56 rect from 0.093950, 40 to 85.128925, 50 fc rgb "#808080"
set object 57 rect from 0.100108, 40 to 85.637191, 50 fc rgb "#800080"
set object 58 rect from 0.100927, 40 to 86.211857, 50 fc rgb "#008080"
set object 59 rect from 0.101382, 40 to 147.762184, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.171157, 50 to 161.591763, 60 fc rgb "#FF0000"
set object 61 rect from 0.033112, 50 to 30.857764, 60 fc rgb "#00FF00"
set object 62 rect from 0.036647, 50 to 34.303247, 60 fc rgb "#0000FF"
set object 63 rect from 0.040402, 50 to 37.806604, 60 fc rgb "#FFFF00"
set object 64 rect from 0.044542, 50 to 41.397675, 60 fc rgb "#FF00FF"
set object 65 rect from 0.048732, 50 to 46.796168, 60 fc rgb "#808080"
set object 66 rect from 0.055091, 50 to 47.395547, 60 fc rgb "#800080"
set object 67 rect from 0.056224, 50 to 48.347377, 60 fc rgb "#008080"
set object 68 rect from 0.056904, 50 to 144.727917, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
