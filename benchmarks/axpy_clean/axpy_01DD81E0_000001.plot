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

set object 15 rect from 0.192042, 0 to 155.346824, 10 fc rgb "#FF0000"
set object 16 rect from 0.058739, 0 to 46.944928, 10 fc rgb "#00FF00"
set object 17 rect from 0.061454, 0 to 47.509734, 10 fc rgb "#0000FF"
set object 18 rect from 0.061957, 0 to 48.162141, 10 fc rgb "#FFFF00"
set object 19 rect from 0.062796, 0 to 48.991291, 10 fc rgb "#FF00FF"
set object 20 rect from 0.063882, 0 to 54.892929, 10 fc rgb "#808080"
set object 21 rect from 0.071572, 0 to 55.302510, 10 fc rgb "#800080"
set object 22 rect from 0.072429, 0 to 55.764344, 10 fc rgb "#008080"
set object 23 rect from 0.072730, 0 to 146.991546, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.189830, 10 to 155.465933, 20 fc rgb "#FF0000"
set object 25 rect from 0.030496, 10 to 25.857325, 20 fc rgb "#00FF00"
set object 26 rect from 0.034094, 10 to 26.973105, 20 fc rgb "#0000FF"
set object 27 rect from 0.035233, 10 to 28.349384, 20 fc rgb "#FFFF00"
set object 28 rect from 0.037049, 10 to 29.450563, 20 fc rgb "#FF00FF"
set object 29 rect from 0.038456, 10 to 35.578893, 20 fc rgb "#808080"
set object 30 rect from 0.046460, 10 to 36.127561, 20 fc rgb "#800080"
set object 31 rect from 0.047597, 10 to 36.746926, 20 fc rgb "#008080"
set object 32 rect from 0.047963, 10 to 144.959015, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.193970, 20 to 161.438268, 30 fc rgb "#FF0000"
set object 34 rect from 0.080889, 20 to 63.690062, 30 fc rgb "#00FF00"
set object 35 rect from 0.083228, 20 to 64.255635, 30 fc rgb "#0000FF"
set object 36 rect from 0.083729, 20 to 66.963627, 30 fc rgb "#FFFF00"
set object 37 rect from 0.087282, 20 to 70.146516, 30 fc rgb "#FF00FF"
set object 38 rect from 0.091411, 20 to 76.031249, 30 fc rgb "#808080"
set object 39 rect from 0.099077, 20 to 76.636783, 30 fc rgb "#800080"
set object 40 rect from 0.100198, 20 to 77.547387, 30 fc rgb "#008080"
set object 41 rect from 0.101034, 20 to 148.446978, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.198860, 30 to 164.030225, 40 fc rgb "#FF0000"
set object 43 rect from 0.164709, 30 to 127.861168, 40 fc rgb "#00FF00"
set object 44 rect from 0.166749, 30 to 128.353739, 40 fc rgb "#0000FF"
set object 45 rect from 0.167146, 30 to 130.816597, 40 fc rgb "#FFFF00"
set object 46 rect from 0.170353, 30 to 133.334015, 40 fc rgb "#FF00FF"
set object 47 rect from 0.173633, 30 to 139.140368, 40 fc rgb "#808080"
set object 48 rect from 0.181188, 30 to 139.632171, 40 fc rgb "#800080"
set object 49 rect from 0.182079, 30 to 140.101689, 40 fc rgb "#008080"
set object 50 rect from 0.182435, 30 to 152.423668, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.197318, 40 to 162.912141, 50 fc rgb "#FF0000"
set object 52 rect from 0.136469, 40 to 106.184938, 50 fc rgb "#00FF00"
set object 53 rect from 0.138539, 40 to 106.751280, 50 fc rgb "#0000FF"
set object 54 rect from 0.139035, 40 to 109.144211, 50 fc rgb "#FFFF00"
set object 55 rect from 0.142239, 40 to 111.821464, 50 fc rgb "#FF00FF"
set object 56 rect from 0.145630, 40 to 117.598617, 50 fc rgb "#808080"
set object 57 rect from 0.153156, 40 to 118.083503, 50 fc rgb "#800080"
set object 58 rect from 0.154066, 40 to 118.573002, 50 fc rgb "#008080"
set object 59 rect from 0.154420, 40 to 151.264087, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.195816, 50 to 162.088372, 60 fc rgb "#FF0000"
set object 61 rect from 0.110058, 50 to 85.963370, 60 fc rgb "#00FF00"
set object 62 rect from 0.112237, 50 to 86.569671, 60 fc rgb "#0000FF"
set object 63 rect from 0.112768, 50 to 89.116290, 60 fc rgb "#FFFF00"
set object 64 rect from 0.116086, 50 to 91.835041, 60 fc rgb "#FF00FF"
set object 65 rect from 0.119627, 50 to 97.636782, 60 fc rgb "#808080"
set object 66 rect from 0.127199, 50 to 98.145492, 60 fc rgb "#800080"
set object 67 rect from 0.128130, 50 to 98.721055, 60 fc rgb "#008080"
set object 68 rect from 0.128590, 50 to 149.963883, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
