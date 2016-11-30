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

set object 15 rect from 0.236948, 0 to 158.793322, 10 fc rgb "#FF0000"
set object 16 rect from 0.205141, 0 to 132.106662, 10 fc rgb "#00FF00"
set object 17 rect from 0.207001, 0 to 132.532447, 10 fc rgb "#0000FF"
set object 18 rect from 0.207421, 0 to 132.955040, 10 fc rgb "#FFFF00"
set object 19 rect from 0.208083, 0 to 133.614829, 10 fc rgb "#FF00FF"
set object 20 rect from 0.209121, 0 to 139.547112, 10 fc rgb "#808080"
set object 21 rect from 0.218402, 0 to 139.889148, 10 fc rgb "#800080"
set object 22 rect from 0.219225, 0 to 140.257402, 10 fc rgb "#008080"
set object 23 rect from 0.219516, 0 to 151.134216, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.235378, 10 to 157.959634, 20 fc rgb "#FF0000"
set object 25 rect from 0.180928, 10 to 116.703434, 20 fc rgb "#00FF00"
set object 26 rect from 0.182909, 10 to 117.165029, 20 fc rgb "#0000FF"
set object 27 rect from 0.183385, 10 to 117.640038, 20 fc rgb "#FFFF00"
set object 28 rect from 0.184176, 10 to 118.405954, 20 fc rgb "#FF00FF"
set object 29 rect from 0.185351, 10 to 124.320975, 20 fc rgb "#808080"
set object 30 rect from 0.194605, 10 to 124.711607, 20 fc rgb "#800080"
set object 31 rect from 0.195485, 10 to 125.105429, 20 fc rgb "#008080"
set object 32 rect from 0.195817, 10 to 149.990463, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.227625, 20 to 160.441492, 30 fc rgb "#FF0000"
set object 34 rect from 0.044039, 20 to 30.753384, 30 fc rgb "#00FF00"
set object 35 rect from 0.048665, 20 to 31.977056, 30 fc rgb "#0000FF"
set object 36 rect from 0.050148, 20 to 35.157047, 30 fc rgb "#FFFF00"
set object 37 rect from 0.055165, 20 to 39.770412, 30 fc rgb "#FF00FF"
set object 38 rect from 0.062341, 20 to 45.766629, 30 fc rgb "#808080"
set object 39 rect from 0.071739, 20 to 46.321557, 30 fc rgb "#800080"
set object 40 rect from 0.073063, 20 to 47.301641, 30 fc rgb "#008080"
set object 41 rect from 0.074153, 20 to 144.894411, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.229794, 30 to 159.143655, 40 fc rgb "#FF0000"
set object 43 rect from 0.085484, 30 to 55.903745, 40 fc rgb "#00FF00"
set object 44 rect from 0.087816, 30 to 56.466990, 40 fc rgb "#0000FF"
set object 45 rect from 0.088442, 30 to 58.572917, 40 fc rgb "#FFFF00"
set object 46 rect from 0.091757, 30 to 62.449146, 40 fc rgb "#FF00FF"
set object 47 rect from 0.097821, 30 to 68.166612, 40 fc rgb "#808080"
set object 48 rect from 0.106778, 30 to 68.651863, 40 fc rgb "#800080"
set object 49 rect from 0.107827, 30 to 69.165874, 40 fc rgb "#008080"
set object 50 rect from 0.108323, 30 to 146.424956, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.231675, 40 to 160.373729, 50 fc rgb "#FF0000"
set object 52 rect from 0.118493, 40 to 76.696475, 50 fc rgb "#00FF00"
set object 53 rect from 0.120328, 40 to 77.078790, 50 fc rgb "#0000FF"
set object 54 rect from 0.120699, 40 to 79.235865, 50 fc rgb "#FFFF00"
set object 55 rect from 0.124062, 40 to 83.260405, 50 fc rgb "#FF00FF"
set object 56 rect from 0.130371, 40 to 88.986189, 50 fc rgb "#808080"
set object 57 rect from 0.139313, 40 to 89.446508, 50 fc rgb "#800080"
set object 58 rect from 0.140345, 40 to 89.930472, 50 fc rgb "#008080"
set object 59 rect from 0.140789, 40 to 147.723421, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.233453, 50 to 161.134510, 60 fc rgb "#FF0000"
set object 61 rect from 0.151264, 50 to 97.709767, 60 fc rgb "#00FF00"
set object 62 rect from 0.153194, 50 to 98.152814, 60 fc rgb "#0000FF"
set object 63 rect from 0.153644, 50 to 100.295828, 60 fc rgb "#FFFF00"
set object 64 rect from 0.157025, 50 to 103.902252, 60 fc rgb "#FF00FF"
set object 65 rect from 0.162642, 50 to 109.633790, 60 fc rgb "#808080"
set object 66 rect from 0.171611, 50 to 110.074922, 60 fc rgb "#800080"
set object 67 rect from 0.172568, 50 to 110.519255, 60 fc rgb "#008080"
set object 68 rect from 0.173003, 50 to 148.917675, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
