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

set object 15 rect from 97.745109, 0 to 228.812162, 10 fc rgb "#FF0000"
set object 16 rect from 7.715449, 0 to 11.371237, 10 fc rgb "#00FF00"
set object 17 rect from 7.718449, 0 to 11.372556, 10 fc rgb "#0000FF"
set object 18 rect from 7.718996, 0 to 11.374172, 10 fc rgb "#FFFF00"
set object 19 rect from 7.720127, 0 to 11.376009, 10 fc rgb "#FF00FF"
set object 20 rect from 7.721374, 0 to 96.170684, 10 fc rgb "#808080"
set object 21 rect from 65.274277, 0 to 96.173120, 10 fc rgb "#800080"
set object 22 rect from 65.276175, 0 to 96.174489, 10 fc rgb "#008080"
set object 23 rect from 65.276563, 0 to 144.010437, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 97.741606, 10 to 213.112752, 20 fc rgb "#FF0000"
set object 25 rect from 15.515568, 10 to 22.864101, 20 fc rgb "#00FF00"
set object 26 rect from 15.518977, 10 to 22.865646, 20 fc rgb "#0000FF"
set object 27 rect from 15.519709, 10 to 22.867503, 20 fc rgb "#FFFF00"
set object 28 rect from 15.521010, 10 to 22.869568, 20 fc rgb "#FF00FF"
set object 29 rect from 15.522390, 10 to 91.969034, 20 fc rgb "#808080"
set object 30 rect from 62.422653, 10 to 91.972051, 20 fc rgb "#800080"
set object 31 rect from 62.424878, 10 to 91.973533, 20 fc rgb "#008080"
set object 32 rect from 62.425345, 10 to 144.004140, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 97.754341, 20 to 280.408789, 30 fc rgb "#FF0000"
set object 34 rect from 3.515042, 20 to 5.182894, 30 fc rgb "#00FF00"
set object 35 rect from 3.518186, 20 to 5.184194, 30 fc rgb "#0000FF"
set object 36 rect from 3.518795, 20 to 5.194363, 30 fc rgb "#FFFF00"
set object 37 rect from 3.525754, 20 to 75.264820, 30 fc rgb "#FF00FF"
set object 38 rect from 51.084625, 20 to 128.151879, 30 fc rgb "#808080"
set object 39 rect from 86.980830, 20 to 141.567927, 30 fc rgb "#800080"
set object 40 rect from 96.087111, 20 to 142.354251, 30 fc rgb "#008080"
set object 41 rect from 96.620222, 20 to 144.024256, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 97.748316, 30 to 269.197138, 40 fc rgb "#FF0000"
set object 43 rect from 0.031940, 30 to 0.055065, 40 fc rgb "#00FF00"
set object 44 rect from 0.037913, 30 to 0.058170, 40 fc rgb "#0000FF"
set object 45 rect from 0.039623, 30 to 0.073113, 40 fc rgb "#FFFF00"
set object 46 rect from 0.049798, 30 to 61.444416, 40 fc rgb "#FF00FF"
set object 47 rect from 41.704245, 30 to 124.231495, 40 fc rgb "#808080"
set object 48 rect from 84.319821, 30 to 125.237211, 40 fc rgb "#800080"
set object 49 rect from 85.003004, 30 to 126.057727, 40 fc rgb "#008080"
set object 50 rect from 85.559329, 30 to 144.015420, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 97.751734, 40 to 256.782433, 50 fc rgb "#FF0000"
set object 52 rect from 13.753921, 40 to 20.269763, 50 fc rgb "#00FF00"
set object 53 rect from 13.758264, 40 to 20.273297, 50 fc rgb "#0000FF"
set object 54 rect from 13.760264, 40 to 20.287562, 50 fc rgb "#FFFF00"
set object 55 rect from 13.770001, 40 to 52.186172, 50 fc rgb "#FF00FF"
set object 56 rect from 35.420507, 40 to 132.042088, 50 fc rgb "#808080"
set object 57 rect from 89.621312, 40 to 133.032881, 50 fc rgb "#800080"
set object 58 rect from 90.294518, 40 to 133.853160, 50 fc rgb "#008080"
set object 59 rect from 90.850337, 40 to 144.020291, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 97.756457, 50 to 270.412968, 60 fc rgb "#FF0000"
set object 61 rect from 11.365482, 50 to 16.750328, 60 fc rgb "#00FF00"
set object 62 rect from 11.369462, 50 to 16.752306, 60 fc rgb "#0000FF"
set object 63 rect from 11.370425, 50 to 16.856436, 60 fc rgb "#FFFF00"
set object 64 rect from 11.441243, 50 to 113.563443, 60 fc rgb "#FF00FF"
set object 65 rect from 77.078983, 50 to 141.053027, 60 fc rgb "#808080"
set object 66 rect from 95.737251, 50 to 143.136800, 60 fc rgb "#800080"
set object 67 rect from 97.151941, 50 to 143.978246, 60 fc rgb "#008080"
set object 68 rect from 97.722524, 50 to 144.027251, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
