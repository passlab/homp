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

set object 15 rect from 0.196206, 0 to 154.046210, 10 fc rgb "#FF0000"
set object 16 rect from 0.139458, 0 to 108.473288, 10 fc rgb "#00FF00"
set object 17 rect from 0.141844, 0 to 109.085937, 10 fc rgb "#0000FF"
set object 18 rect from 0.142396, 0 to 109.691684, 10 fc rgb "#FFFF00"
set object 19 rect from 0.143177, 0 to 110.516728, 10 fc rgb "#FF00FF"
set object 20 rect from 0.144265, 0 to 112.229690, 10 fc rgb "#808080"
set object 21 rect from 0.146506, 0 to 112.672115, 10 fc rgb "#800080"
set object 22 rect from 0.147319, 0 to 113.121442, 10 fc rgb "#008080"
set object 23 rect from 0.147648, 0 to 149.964697, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.192736, 10 to 151.910758, 20 fc rgb "#FF0000"
set object 25 rect from 0.055555, 10 to 44.320786, 20 fc rgb "#00FF00"
set object 26 rect from 0.058173, 10 to 44.948771, 20 fc rgb "#0000FF"
set object 27 rect from 0.058735, 10 to 45.730875, 20 fc rgb "#FFFF00"
set object 28 rect from 0.059791, 10 to 46.710806, 20 fc rgb "#FF00FF"
set object 29 rect from 0.061068, 10 to 48.587089, 20 fc rgb "#808080"
set object 30 rect from 0.063536, 10 to 49.081656, 20 fc rgb "#800080"
set object 31 rect from 0.064406, 10 to 49.531749, 20 fc rgb "#008080"
set object 32 rect from 0.064733, 10 to 147.139152, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.190461, 20 to 153.262568, 30 fc rgb "#FF0000"
set object 34 rect from 0.028025, 20 to 24.005239, 30 fc rgb "#00FF00"
set object 35 rect from 0.031746, 20 to 24.938397, 30 fc rgb "#0000FF"
set object 36 rect from 0.032650, 20 to 27.868989, 30 fc rgb "#FFFF00"
set object 37 rect from 0.036517, 20 to 29.561248, 30 fc rgb "#FF00FF"
set object 38 rect from 0.038694, 20 to 31.296447, 30 fc rgb "#808080"
set object 39 rect from 0.041059, 20 to 31.989605, 30 fc rgb "#800080"
set object 40 rect from 0.042258, 20 to 32.924297, 30 fc rgb "#008080"
set object 41 rect from 0.043079, 20 to 144.995266, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.199203, 30 to 157.247471, 40 fc rgb "#FF0000"
set object 43 rect from 0.173169, 30 to 134.015133, 40 fc rgb "#00FF00"
set object 44 rect from 0.175206, 30 to 134.623181, 40 fc rgb "#0000FF"
set object 45 rect from 0.175685, 30 to 136.053972, 40 fc rgb "#FFFF00"
set object 46 rect from 0.177562, 30 to 137.132816, 40 fc rgb "#FF00FF"
set object 47 rect from 0.178959, 30 to 138.641051, 40 fc rgb "#808080"
set object 48 rect from 0.180926, 30 to 139.119515, 40 fc rgb "#800080"
set object 49 rect from 0.181804, 30 to 139.581877, 40 fc rgb "#008080"
set object 50 rect from 0.182155, 30 to 152.393054, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.194454, 40 to 154.740133, 50 fc rgb "#FF0000"
set object 52 rect from 0.115823, 40 to 90.495621, 50 fc rgb "#00FF00"
set object 53 rect from 0.118463, 40 to 91.143541, 50 fc rgb "#0000FF"
set object 54 rect from 0.118981, 40 to 93.262890, 50 fc rgb "#FFFF00"
set object 55 rect from 0.121760, 40 to 94.566397, 50 fc rgb "#FF00FF"
set object 56 rect from 0.123489, 40 to 96.241020, 50 fc rgb "#808080"
set object 57 rect from 0.125657, 40 to 96.801529, 50 fc rgb "#800080"
set object 58 rect from 0.126639, 40 to 97.441014, 50 fc rgb "#008080"
set object 59 rect from 0.127205, 40 to 148.585279, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.197761, 50 to 156.224600, 60 fc rgb "#FF0000"
set object 61 rect from 0.155435, 50 to 120.505428, 60 fc rgb "#00FF00"
set object 62 rect from 0.157535, 50 to 121.093540, 60 fc rgb "#0000FF"
set object 63 rect from 0.158066, 50 to 122.674618, 60 fc rgb "#FFFF00"
set object 64 rect from 0.160110, 50 to 123.669118, 60 fc rgb "#FF00FF"
set object 65 rect from 0.161421, 50 to 125.211090, 60 fc rgb "#808080"
set object 66 rect from 0.163501, 50 to 125.753196, 60 fc rgb "#800080"
set object 67 rect from 0.164373, 50 to 126.293769, 60 fc rgb "#008080"
set object 68 rect from 0.164841, 50 to 151.223732, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
