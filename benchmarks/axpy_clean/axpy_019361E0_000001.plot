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

set object 15 rect from 0.202967, 0 to 157.260992, 10 fc rgb "#FF0000"
set object 16 rect from 0.122461, 0 to 91.922855, 10 fc rgb "#00FF00"
set object 17 rect from 0.124927, 0 to 92.463096, 10 fc rgb "#0000FF"
set object 18 rect from 0.125413, 0 to 93.063855, 10 fc rgb "#FFFF00"
set object 19 rect from 0.126223, 0 to 93.882335, 10 fc rgb "#FF00FF"
set object 20 rect from 0.127326, 0 to 99.543054, 10 fc rgb "#808080"
set object 21 rect from 0.135006, 0 to 99.947497, 10 fc rgb "#800080"
set object 22 rect from 0.135812, 0 to 100.374081, 10 fc rgb "#008080"
set object 23 rect from 0.136115, 0 to 149.262919, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.197293, 10 to 155.221804, 20 fc rgb "#FF0000"
set object 25 rect from 0.035359, 10 to 28.781845, 20 fc rgb "#00FF00"
set object 26 rect from 0.039439, 10 to 29.765644, 20 fc rgb "#0000FF"
set object 27 rect from 0.040461, 10 to 31.308134, 20 fc rgb "#FFFF00"
set object 28 rect from 0.042576, 10 to 32.593789, 20 fc rgb "#FF00FF"
set object 29 rect from 0.044292, 10 to 38.538652, 20 fc rgb "#808080"
set object 30 rect from 0.052372, 10 to 39.034611, 20 fc rgb "#800080"
set object 31 rect from 0.053425, 10 to 39.624300, 20 fc rgb "#008080"
set object 32 rect from 0.053824, 10 to 144.843573, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.204605, 20 to 162.123897, 30 fc rgb "#FF0000"
set object 34 rect from 0.144873, 20 to 108.225285, 30 fc rgb "#00FF00"
set object 35 rect from 0.146995, 20 to 108.716077, 30 fc rgb "#0000FF"
set object 36 rect from 0.147419, 20 to 111.125020, 30 fc rgb "#FFFF00"
set object 37 rect from 0.150703, 20 to 113.764229, 30 fc rgb "#FF00FF"
set object 38 rect from 0.154275, 20 to 119.411664, 30 fc rgb "#808080"
set object 39 rect from 0.161923, 20 to 119.894338, 30 fc rgb "#800080"
set object 40 rect from 0.162821, 20 to 120.374798, 30 fc rgb "#008080"
set object 41 rect from 0.163227, 20 to 150.588428, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.206099, 30 to 163.134265, 40 fc rgb "#FF0000"
set object 43 rect from 0.172175, 30 to 128.320471, 40 fc rgb "#00FF00"
set object 44 rect from 0.174228, 30 to 128.899089, 40 fc rgb "#0000FF"
set object 45 rect from 0.174765, 30 to 131.437925, 40 fc rgb "#FFFF00"
set object 46 rect from 0.178267, 30 to 133.795205, 40 fc rgb "#FF00FF"
set object 47 rect from 0.181403, 30 to 139.451497, 40 fc rgb "#808080"
set object 48 rect from 0.189073, 30 to 139.919410, 40 fc rgb "#800080"
set object 49 rect from 0.189985, 30 to 140.410941, 40 fc rgb "#008080"
set object 50 rect from 0.190362, 30 to 151.753783, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.201113, 40 to 159.332652, 50 fc rgb "#FF0000"
set object 52 rect from 0.094797, 40 to 71.400347, 50 fc rgb "#00FF00"
set object 53 rect from 0.097100, 40 to 71.933946, 50 fc rgb "#0000FF"
set object 54 rect from 0.097579, 40 to 74.093433, 50 fc rgb "#FFFF00"
set object 55 rect from 0.100506, 40 to 76.672861, 50 fc rgb "#FF00FF"
set object 56 rect from 0.104005, 40 to 82.329891, 50 fc rgb "#808080"
set object 57 rect from 0.111672, 40 to 82.819945, 50 fc rgb "#800080"
set object 58 rect from 0.112587, 40 to 83.270146, 50 fc rgb "#008080"
set object 59 rect from 0.112947, 40 to 147.926340, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.199243, 50 to 159.307560, 60 fc rgb "#FF0000"
set object 61 rect from 0.065166, 50 to 49.688867, 60 fc rgb "#00FF00"
set object 62 rect from 0.067689, 50 to 50.318410, 60 fc rgb "#0000FF"
set object 63 rect from 0.068311, 50 to 52.899315, 60 fc rgb "#FFFF00"
set object 64 rect from 0.071842, 50 to 56.371764, 60 fc rgb "#FF00FF"
set object 65 rect from 0.076528, 50 to 61.932111, 60 fc rgb "#808080"
set object 66 rect from 0.084072, 50 to 62.571249, 60 fc rgb "#800080"
set object 67 rect from 0.085191, 50 to 63.329209, 60 fc rgb "#008080"
set object 68 rect from 0.085938, 50 to 146.538098, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
