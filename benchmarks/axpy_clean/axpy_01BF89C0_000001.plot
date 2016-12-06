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

set object 15 rect from 0.139051, 0 to 159.065491, 10 fc rgb "#FF0000"
set object 16 rect from 0.118228, 0 to 133.860607, 10 fc rgb "#00FF00"
set object 17 rect from 0.120142, 0 to 134.560181, 10 fc rgb "#0000FF"
set object 18 rect from 0.120527, 0 to 135.318989, 10 fc rgb "#FFFF00"
set object 19 rect from 0.121208, 0 to 136.494636, 10 fc rgb "#FF00FF"
set object 20 rect from 0.122260, 0 to 137.752979, 10 fc rgb "#808080"
set object 21 rect from 0.123388, 0 to 138.340798, 10 fc rgb "#800080"
set object 22 rect from 0.124173, 0 to 138.926394, 10 fc rgb "#008080"
set object 23 rect from 0.124456, 0 to 154.724990, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.137451, 10 to 157.556818, 20 fc rgb "#FF0000"
set object 25 rect from 0.102391, 10 to 116.254949, 20 fc rgb "#00FF00"
set object 26 rect from 0.104394, 10 to 117.043931, 20 fc rgb "#0000FF"
set object 27 rect from 0.104854, 10 to 117.864203, 20 fc rgb "#FFFF00"
set object 28 rect from 0.105619, 10 to 119.189598, 20 fc rgb "#FF00FF"
set object 29 rect from 0.106802, 10 to 120.411065, 20 fc rgb "#808080"
set object 30 rect from 0.107869, 10 to 121.086060, 20 fc rgb "#800080"
set object 31 rect from 0.108739, 10 to 121.774461, 20 fc rgb "#008080"
set object 32 rect from 0.109090, 10 to 152.912340, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.135761, 20 to 155.453623, 30 fc rgb "#FF0000"
set object 34 rect from 0.087516, 20 to 99.423752, 30 fc rgb "#00FF00"
set object 35 rect from 0.089343, 20 to 100.172502, 30 fc rgb "#0000FF"
set object 36 rect from 0.089759, 20 to 101.003948, 30 fc rgb "#FFFF00"
set object 37 rect from 0.090499, 20 to 102.314822, 30 fc rgb "#FF00FF"
set object 38 rect from 0.091675, 20 to 103.316133, 30 fc rgb "#808080"
set object 39 rect from 0.092571, 20 to 103.984417, 30 fc rgb "#800080"
set object 40 rect from 0.093427, 20 to 104.667231, 30 fc rgb "#008080"
set object 41 rect from 0.093803, 20 to 151.205868, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.134275, 30 to 153.877906, 40 fc rgb "#FF0000"
set object 43 rect from 0.071295, 30 to 81.355444, 40 fc rgb "#00FF00"
set object 44 rect from 0.073180, 30 to 82.109781, 40 fc rgb "#0000FF"
set object 45 rect from 0.073596, 30 to 83.063041, 40 fc rgb "#FFFF00"
set object 46 rect from 0.074454, 30 to 84.402965, 40 fc rgb "#FF00FF"
set object 47 rect from 0.075643, 30 to 85.403160, 40 fc rgb "#808080"
set object 48 rect from 0.076540, 30 to 86.013336, 40 fc rgb "#800080"
set object 49 rect from 0.077350, 30 to 86.730787, 40 fc rgb "#008080"
set object 50 rect from 0.077731, 30 to 149.408872, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.132702, 40 to 152.448558, 50 fc rgb "#FF0000"
set object 52 rect from 0.055179, 40 to 63.821318, 50 fc rgb "#00FF00"
set object 53 rect from 0.057498, 40 to 64.677352, 50 fc rgb "#0000FF"
set object 54 rect from 0.057995, 40 to 65.661893, 50 fc rgb "#FFFF00"
set object 55 rect from 0.058891, 40 to 67.056580, 50 fc rgb "#FF00FF"
set object 56 rect from 0.060148, 40 to 68.147298, 50 fc rgb "#808080"
set object 57 rect from 0.061120, 40 to 68.853575, 50 fc rgb "#800080"
set object 58 rect from 0.062039, 40 to 69.687262, 50 fc rgb "#008080"
set object 59 rect from 0.062478, 40 to 147.539238, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.130883, 50 to 153.301261, 60 fc rgb "#FF0000"
set object 61 rect from 0.033619, 50 to 40.703983, 60 fc rgb "#00FF00"
set object 62 rect from 0.036957, 50 to 42.218243, 60 fc rgb "#0000FF"
set object 63 rect from 0.037932, 50 to 44.521486, 60 fc rgb "#FFFF00"
set object 64 rect from 0.040007, 50 to 46.461527, 60 fc rgb "#FF00FF"
set object 65 rect from 0.041714, 50 to 47.750045, 60 fc rgb "#808080"
set object 66 rect from 0.042882, 50 to 48.868698, 60 fc rgb "#800080"
set object 67 rect from 0.044418, 50 to 50.328196, 60 fc rgb "#008080"
set object 68 rect from 0.045183, 50 to 145.194647, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0