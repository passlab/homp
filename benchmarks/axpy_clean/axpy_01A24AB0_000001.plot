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

set object 15 rect from 0.150844, 0 to 156.423820, 10 fc rgb "#FF0000"
set object 16 rect from 0.114665, 0 to 118.457053, 10 fc rgb "#00FF00"
set object 17 rect from 0.116772, 0 to 119.295996, 10 fc rgb "#0000FF"
set object 18 rect from 0.117288, 0 to 120.100289, 10 fc rgb "#FFFF00"
set object 19 rect from 0.118080, 0 to 121.123466, 10 fc rgb "#FF00FF"
set object 20 rect from 0.119082, 0 to 121.548004, 10 fc rgb "#808080"
set object 21 rect from 0.119504, 0 to 122.060108, 10 fc rgb "#800080"
set object 22 rect from 0.120262, 0 to 122.629194, 10 fc rgb "#008080"
set object 23 rect from 0.120568, 0 to 153.000933, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.149172, 10 to 155.035150, 20 fc rgb "#FF0000"
set object 25 rect from 0.099325, 10 to 103.139865, 20 fc rgb "#00FF00"
set object 26 rect from 0.101674, 10 to 103.932992, 20 fc rgb "#0000FF"
set object 27 rect from 0.102196, 10 to 104.743354, 20 fc rgb "#FFFF00"
set object 28 rect from 0.103040, 10 to 105.969151, 20 fc rgb "#FF00FF"
set object 29 rect from 0.104217, 10 to 106.446666, 20 fc rgb "#808080"
set object 30 rect from 0.104688, 10 to 107.084991, 20 fc rgb "#800080"
set object 31 rect from 0.105547, 10 to 107.679563, 20 fc rgb "#008080"
set object 32 rect from 0.105890, 10 to 151.209084, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.145689, 20 to 151.496176, 30 fc rgb "#FF0000"
set object 34 rect from 0.067875, 20 to 71.256269, 30 fc rgb "#00FF00"
set object 35 rect from 0.070358, 20 to 72.061593, 30 fc rgb "#0000FF"
set object 36 rect from 0.070891, 20 to 72.960552, 30 fc rgb "#FFFF00"
set object 37 rect from 0.071797, 20 to 74.162926, 30 fc rgb "#FF00FF"
set object 38 rect from 0.072974, 20 to 74.598690, 30 fc rgb "#808080"
set object 39 rect from 0.073399, 20 to 75.175907, 30 fc rgb "#800080"
set object 40 rect from 0.074217, 20 to 75.795967, 30 fc rgb "#008080"
set object 41 rect from 0.074593, 20 to 147.726242, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.143894, 30 to 154.099417, 40 fc rgb "#FF0000"
set object 43 rect from 0.042831, 30 to 48.778847, 40 fc rgb "#00FF00"
set object 44 rect from 0.048420, 30 to 50.555403, 40 fc rgb "#0000FF"
set object 45 rect from 0.049788, 30 to 53.065011, 40 fc rgb "#FFFF00"
set object 46 rect from 0.052290, 30 to 55.057417, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054218, 30 to 56.536688, 40 fc rgb "#808080"
set object 48 rect from 0.055732, 30 to 57.405122, 40 fc rgb "#800080"
set object 49 rect from 0.056933, 30 to 58.612594, 40 fc rgb "#008080"
set object 50 rect from 0.057789, 30 to 145.397834, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.152408, 40 to 158.134172, 50 fc rgb "#FF0000"
set object 52 rect from 0.130095, 40 to 134.273116, 50 fc rgb "#00FF00"
set object 53 rect from 0.132233, 40 to 134.986809, 50 fc rgb "#0000FF"
set object 54 rect from 0.132700, 40 to 135.835886, 50 fc rgb "#FFFF00"
set object 55 rect from 0.133547, 40 to 136.959918, 50 fc rgb "#FF00FF"
set object 56 rect from 0.134650, 40 to 137.385427, 50 fc rgb "#808080"
set object 57 rect from 0.135072, 40 to 137.958639, 50 fc rgb "#800080"
set object 58 rect from 0.135872, 40 to 138.592899, 50 fc rgb "#008080"
set object 59 rect from 0.136247, 40 to 154.632943, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.147431, 50 to 153.090562, 60 fc rgb "#FF0000"
set object 61 rect from 0.083971, 50 to 87.355357, 60 fc rgb "#00FF00"
set object 62 rect from 0.086175, 50 to 88.090411, 60 fc rgb "#0000FF"
set object 63 rect from 0.086639, 50 to 88.993496, 60 fc rgb "#FFFF00"
set object 64 rect from 0.087538, 50 to 90.145988, 60 fc rgb "#FF00FF"
set object 65 rect from 0.088659, 50 to 90.556266, 60 fc rgb "#808080"
set object 66 rect from 0.089078, 50 to 91.106055, 60 fc rgb "#800080"
set object 67 rect from 0.089858, 50 to 91.706695, 60 fc rgb "#008080"
set object 68 rect from 0.090210, 50 to 149.416265, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
