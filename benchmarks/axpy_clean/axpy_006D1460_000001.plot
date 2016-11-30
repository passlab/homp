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

set object 15 rect from 0.266741, 0 to 157.034218, 10 fc rgb "#FF0000"
set object 16 rect from 0.167900, 0 to 94.087894, 10 fc rgb "#00FF00"
set object 17 rect from 0.169930, 0 to 94.622848, 10 fc rgb "#0000FF"
set object 18 rect from 0.170620, 0 to 95.057360, 10 fc rgb "#FFFF00"
set object 19 rect from 0.171431, 0 to 95.648916, 10 fc rgb "#FF00FF"
set object 20 rect from 0.172494, 0 to 103.222050, 10 fc rgb "#808080"
set object 21 rect from 0.186153, 0 to 103.535031, 10 fc rgb "#800080"
set object 22 rect from 0.186982, 0 to 103.874094, 10 fc rgb "#008080"
set object 23 rect from 0.187303, 0 to 147.709712, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.269764, 10 to 158.307768, 20 fc rgb "#FF0000"
set object 25 rect from 0.236191, 10 to 131.815950, 20 fc rgb "#00FF00"
set object 26 rect from 0.237880, 10 to 132.170550, 20 fc rgb "#0000FF"
set object 27 rect from 0.238289, 10 to 132.530698, 20 fc rgb "#FFFF00"
set object 28 rect from 0.238943, 10 to 133.065098, 20 fc rgb "#FF00FF"
set object 29 rect from 0.239902, 10 to 140.507824, 20 fc rgb "#808080"
set object 30 rect from 0.253336, 10 to 140.820805, 20 fc rgb "#800080"
set object 31 rect from 0.254165, 10 to 141.134341, 20 fc rgb "#008080"
set object 32 rect from 0.254467, 10 to 149.360073, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.263494, 20 to 160.524166, 30 fc rgb "#FF0000"
set object 34 rect from 0.090856, 20 to 51.386554, 30 fc rgb "#00FF00"
set object 35 rect from 0.092963, 20 to 51.829943, 30 fc rgb "#0000FF"
set object 36 rect from 0.093511, 20 to 53.453115, 30 fc rgb "#FFFF00"
set object 37 rect from 0.096448, 20 to 58.298213, 30 fc rgb "#FF00FF"
set object 38 rect from 0.105175, 20 to 65.588891, 30 fc rgb "#808080"
set object 39 rect from 0.118304, 20 to 66.091657, 30 fc rgb "#800080"
set object 40 rect from 0.119499, 20 to 66.491207, 30 fc rgb "#008080"
set object 41 rect from 0.119960, 20 to 145.856799, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.261540, 30 to 161.536354, 40 fc rgb "#FF0000"
set object 43 rect from 0.045640, 30 to 26.982379, 40 fc rgb "#00FF00"
set object 44 rect from 0.049135, 30 to 27.795906, 40 fc rgb "#0000FF"
set object 45 rect from 0.050207, 30 to 30.243707, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054649, 30 to 35.913429, 40 fc rgb "#FF00FF"
set object 47 rect from 0.064852, 30 to 43.345612, 40 fc rgb "#808080"
set object 48 rect from 0.078240, 30 to 43.896661, 40 fc rgb "#800080"
set object 49 rect from 0.079630, 30 to 44.564796, 40 fc rgb "#008080"
set object 50 rect from 0.080446, 30 to 144.623743, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.268235, 40 to 162.812141, 50 fc rgb "#FF0000"
set object 52 rect from 0.199872, 40 to 111.661985, 50 fc rgb "#00FF00"
set object 53 rect from 0.201580, 40 to 112.010482, 50 fc rgb "#0000FF"
set object 54 rect from 0.201964, 40 to 113.593698, 50 fc rgb "#FFFF00"
set object 55 rect from 0.204823, 40 to 118.264549, 50 fc rgb "#FF00FF"
set object 56 rect from 0.213254, 40 to 125.553560, 50 fc rgb "#808080"
set object 57 rect from 0.226372, 40 to 126.041896, 50 fc rgb "#800080"
set object 58 rect from 0.227507, 40 to 126.378185, 50 fc rgb "#008080"
set object 59 rect from 0.227857, 40 to 148.569852, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.264994, 50 to 161.932573, 60 fc rgb "#FF0000"
set object 61 rect from 0.128853, 50 to 72.370692, 60 fc rgb "#00FF00"
set object 62 rect from 0.130752, 50 to 72.824070, 60 fc rgb "#0000FF"
set object 63 rect from 0.131338, 50 to 74.433371, 60 fc rgb "#FFFF00"
set object 64 rect from 0.134242, 50 to 79.796772, 60 fc rgb "#FF00FF"
set object 65 rect from 0.143918, 50 to 87.176235, 60 fc rgb "#808080"
set object 66 rect from 0.157216, 50 to 87.634054, 60 fc rgb "#800080"
set object 67 rect from 0.158337, 50 to 87.986992, 60 fc rgb "#008080"
set object 68 rect from 0.158676, 50 to 146.751901, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
