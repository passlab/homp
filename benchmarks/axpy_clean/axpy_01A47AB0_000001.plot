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

set object 15 rect from 91.205209, 0 to 166.325121, 10 fc rgb "#FF0000"
set object 16 rect from 0.036892, 0 to 0.065724, 10 fc rgb "#00FF00"
set object 17 rect from 0.042036, 0 to 0.067268, 10 fc rgb "#0000FF"
set object 18 rect from 0.042717, 0 to 0.069471, 10 fc rgb "#FFFF00"
set object 19 rect from 0.044143, 0 to 0.071642, 10 fc rgb "#FF00FF"
set object 20 rect from 0.045487, 0 to 22.384297, 10 fc rgb "#808080"
set object 21 rect from 14.177477, 0 to 22.386799, 10 fc rgb "#800080"
set object 22 rect from 14.179239, 0 to 22.388555, 10 fc rgb "#008080"
set object 23 rect from 14.179773, 0 to 144.003532, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 91.208161, 10 to 203.337844, 20 fc rgb "#FF0000"
set object 25 rect from 14.457049, 10 to 22.831608, 20 fc rgb "#00FF00"
set object 26 rect from 14.460717, 10 to 22.833093, 20 fc rgb "#0000FF"
set object 27 rect from 14.461309, 10 to 22.834825, 20 fc rgb "#FFFF00"
set object 28 rect from 14.462417, 10 to 22.836730, 20 fc rgb "#FF00FF"
set object 29 rect from 14.463602, 10 to 82.158135, 20 fc rgb "#808080"
set object 30 rect from 52.034869, 10 to 82.160892, 20 fc rgb "#800080"
set object 31 rect from 52.036733, 10 to 82.162579, 20 fc rgb "#008080"
set object 32 rect from 52.037317, 10 to 144.009290, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 91.214387, 20 to 269.405877, 30 fc rgb "#FF0000"
set object 34 rect from 9.122206, 20 to 14.408516, 30 fc rgb "#00FF00"
set object 35 rect from 9.125953, 20 to 14.410479, 30 fc rgb "#0000FF"
set object 36 rect from 9.126898, 20 to 14.514694, 30 fc rgb "#FFFF00"
set object 37 rect from 9.192942, 20 to 92.524151, 30 fc rgb "#FF00FF"
set object 38 rect from 58.599782, 20 to 138.784672, 30 fc rgb "#808080"
set object 39 rect from 87.898803, 20 to 139.795972, 30 fc rgb "#800080"
set object 40 rect from 88.539691, 20 to 143.988843, 30 fc rgb "#008080"
set object 41 rect from 91.194558, 20 to 144.019257, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 91.210847, 30 to 245.454556, 40 fc rgb "#FF0000"
set object 43 rect from 11.831384, 30 to 18.685325, 40 fc rgb "#00FF00"
set object 44 rect from 11.834861, 30 to 18.687502, 40 fc rgb "#0000FF"
set object 45 rect from 11.835715, 30 to 26.720899, 40 fc rgb "#FFFF00"
set object 46 rect from 16.924588, 30 to 99.242623, 40 fc rgb "#FF00FF"
set object 47 rect from 62.854808, 30 to 119.058412, 40 fc rgb "#808080"
set object 48 rect from 75.405372, 30 to 120.128834, 40 fc rgb "#800080"
set object 49 rect from 76.083712, 30 to 120.902536, 40 fc rgb "#008080"
set object 50 rect from 76.573172, 30 to 144.013407, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 91.212736, 40 to 258.764443, 50 fc rgb "#FF0000"
set object 52 rect from 16.895973, 40 to 26.681200, 50 fc rgb "#00FF00"
set object 53 rect from 16.898739, 40 to 26.682316, 50 fc rgb "#0000FF"
set object 54 rect from 16.899170, 40 to 26.752236, 50 fc rgb "#FFFF00"
set object 55 rect from 16.943495, 40 to 105.920244, 50 fc rgb "#FF00FF"
set object 56 rect from 67.084098, 40 to 126.247226, 50 fc rgb "#808080"
set object 57 rect from 79.958374, 40 to 141.429836, 50 fc rgb "#800080"
set object 58 rect from 89.574394, 40 to 143.222008, 50 fc rgb "#008080"
set object 59 rect from 90.708931, 40 to 144.016681, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 91.215934, 50 to 279.328463, 60 fc rgb "#FF0000"
set object 61 rect from 4.308399, 50 to 6.807125, 60 fc rgb "#00FF00"
set object 62 rect from 4.311729, 50 to 6.809343, 60 fc rgb "#0000FF"
set object 63 rect from 4.312763, 50 to 6.821477, 60 fc rgb "#FFFF00"
set object 64 rect from 4.320491, 50 to 112.792369, 60 fc rgb "#FF00FF"
set object 65 rect from 71.436441, 50 to 133.596604, 60 fc rgb "#808080"
set object 66 rect from 84.612991, 50 to 142.114708, 60 fc rgb "#800080"
set object 67 rect from 90.008176, 50 to 143.982529, 60 fc rgb "#008080"
set object 68 rect from 91.190691, 50 to 144.021775, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
