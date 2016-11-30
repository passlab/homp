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

set object 15 rect from 0.134525, 0 to 156.374837, 10 fc rgb "#FF0000"
set object 16 rect from 0.085035, 0 to 98.657440, 10 fc rgb "#00FF00"
set object 17 rect from 0.087472, 0 to 99.378934, 10 fc rgb "#0000FF"
set object 18 rect from 0.087856, 0 to 100.189877, 10 fc rgb "#FFFF00"
set object 19 rect from 0.088593, 0 to 101.417616, 10 fc rgb "#FF00FF"
set object 20 rect from 0.089680, 0 to 102.894528, 10 fc rgb "#808080"
set object 21 rect from 0.090980, 0 to 103.529916, 10 fc rgb "#800080"
set object 22 rect from 0.091786, 0 to 104.140359, 10 fc rgb "#008080"
set object 23 rect from 0.092068, 0 to 151.639369, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.129247, 10 to 153.185376, 20 fc rgb "#FF0000"
set object 25 rect from 0.029842, 10 to 37.218414, 20 fc rgb "#00FF00"
set object 26 rect from 0.033391, 10 to 38.933158, 20 fc rgb "#0000FF"
set object 27 rect from 0.034510, 10 to 40.707950, 20 fc rgb "#FFFF00"
set object 28 rect from 0.036116, 10 to 42.532563, 20 fc rgb "#FF00FF"
set object 29 rect from 0.037715, 10 to 44.386643, 20 fc rgb "#808080"
set object 30 rect from 0.039346, 10 to 45.178313, 20 fc rgb "#800080"
set object 31 rect from 0.040485, 10 to 46.101387, 20 fc rgb "#008080"
set object 32 rect from 0.040830, 10 to 145.231114, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137584, 20 to 159.967390, 30 fc rgb "#FF0000"
set object 34 rect from 0.115632, 20 to 132.927667, 30 fc rgb "#00FF00"
set object 35 rect from 0.117719, 20 to 133.599307, 30 fc rgb "#0000FF"
set object 36 rect from 0.118075, 20 to 134.573350, 30 fc rgb "#FFFF00"
set object 37 rect from 0.118934, 20 to 135.825999, 30 fc rgb "#FF00FF"
set object 38 rect from 0.120042, 20 to 137.173802, 30 fc rgb "#808080"
set object 39 rect from 0.121234, 20 to 137.843180, 30 fc rgb "#800080"
set object 40 rect from 0.122086, 20 to 138.608791, 30 fc rgb "#008080"
set object 41 rect from 0.122502, 20 to 155.303346, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.132737, 30 to 154.836729, 40 fc rgb "#FF0000"
set object 43 rect from 0.068334, 30 to 79.643369, 40 fc rgb "#00FF00"
set object 44 rect from 0.070685, 30 to 80.353488, 40 fc rgb "#0000FF"
set object 45 rect from 0.071076, 30 to 81.371715, 40 fc rgb "#FFFF00"
set object 46 rect from 0.071979, 30 to 82.821421, 40 fc rgb "#FF00FF"
set object 47 rect from 0.073254, 30 to 84.230386, 40 fc rgb "#808080"
set object 48 rect from 0.074502, 30 to 84.996031, 40 fc rgb "#800080"
set object 49 rect from 0.075434, 30 to 85.820543, 40 fc rgb "#008080"
set object 50 rect from 0.075895, 30 to 149.622222, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.136074, 40 to 158.363697, 50 fc rgb "#FF0000"
set object 52 rect from 0.100319, 40 to 115.601168, 50 fc rgb "#00FF00"
set object 53 rect from 0.102422, 40 to 116.279592, 50 fc rgb "#0000FF"
set object 54 rect from 0.102786, 40 to 117.268352, 50 fc rgb "#FFFF00"
set object 55 rect from 0.103666, 40 to 118.595766, 50 fc rgb "#FF00FF"
set object 56 rect from 0.104842, 40 to 119.963922, 50 fc rgb "#808080"
set object 57 rect from 0.106047, 40 to 120.674076, 50 fc rgb "#800080"
set object 58 rect from 0.106954, 40 to 121.499735, 50 fc rgb "#008080"
set object 59 rect from 0.107394, 40 to 153.516098, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.131037, 50 to 153.417536, 60 fc rgb "#FF0000"
set object 61 rect from 0.050867, 50 to 60.172807, 60 fc rgb "#00FF00"
set object 62 rect from 0.053492, 50 to 61.039242, 60 fc rgb "#0000FF"
set object 63 rect from 0.054008, 50 to 62.223944, 60 fc rgb "#FFFF00"
set object 64 rect from 0.055093, 50 to 63.857171, 60 fc rgb "#FF00FF"
set object 65 rect from 0.056532, 50 to 65.266136, 60 fc rgb "#808080"
set object 66 rect from 0.057772, 50 to 66.071408, 60 fc rgb "#800080"
set object 67 rect from 0.058735, 50 to 67.536979, 60 fc rgb "#008080"
set object 68 rect from 0.059758, 50 to 147.654897, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
