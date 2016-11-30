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

set object 15 rect from 0.260009, 0 to 153.867884, 10 fc rgb "#FF0000"
set object 16 rect from 0.112154, 0 to 65.022891, 10 fc rgb "#00FF00"
set object 17 rect from 0.115016, 0 to 65.508940, 10 fc rgb "#0000FF"
set object 18 rect from 0.115619, 0 to 66.002930, 10 fc rgb "#FFFF00"
set object 19 rect from 0.116553, 0 to 66.743064, 10 fc rgb "#FF00FF"
set object 20 rect from 0.117812, 0 to 71.489563, 10 fc rgb "#808080"
set object 21 rect from 0.126196, 0 to 71.892240, 10 fc rgb "#800080"
set object 22 rect from 0.127162, 0 to 72.241607, 10 fc rgb "#008080"
set object 23 rect from 0.127514, 0 to 146.620245, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.267752, 10 to 157.629806, 20 fc rgb "#FF0000"
set object 25 rect from 0.232756, 10 to 133.053959, 20 fc rgb "#00FF00"
set object 26 rect from 0.234972, 10 to 133.521292, 20 fc rgb "#0000FF"
set object 27 rect from 0.235536, 10 to 133.978984, 20 fc rgb "#FFFF00"
set object 28 rect from 0.236380, 10 to 134.619866, 20 fc rgb "#FF00FF"
set object 29 rect from 0.237474, 10 to 138.940434, 20 fc rgb "#808080"
set object 30 rect from 0.245106, 10 to 139.257472, 20 fc rgb "#800080"
set object 31 rect from 0.245913, 10 to 139.572241, 20 fc rgb "#008080"
set object 32 rect from 0.246211, 10 to 151.523277, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.264318, 20 to 159.630722, 30 fc rgb "#FF0000"
set object 34 rect from 0.171811, 20 to 98.460496, 30 fc rgb "#00FF00"
set object 35 rect from 0.173993, 20 to 98.973769, 30 fc rgb "#0000FF"
set object 36 rect from 0.174624, 20 to 101.106262, 30 fc rgb "#FFFF00"
set object 37 rect from 0.178393, 20 to 103.964144, 30 fc rgb "#FF00FF"
set object 38 rect from 0.183427, 20 to 108.246144, 30 fc rgb "#808080"
set object 39 rect from 0.190998, 20 to 108.617629, 30 fc rgb "#800080"
set object 40 rect from 0.191903, 20 to 108.988547, 30 fc rgb "#008080"
set object 41 rect from 0.192284, 20 to 149.524065, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.256899, 30 to 158.922347, 40 fc rgb "#FF0000"
set object 43 rect from 0.064669, 30 to 39.972903, 40 fc rgb "#00FF00"
set object 44 rect from 0.070939, 30 to 41.335769, 40 fc rgb "#0000FF"
set object 45 rect from 0.073052, 30 to 44.582717, 40 fc rgb "#FFFF00"
set object 46 rect from 0.078890, 30 to 48.384340, 40 fc rgb "#FF00FF"
set object 47 rect from 0.085464, 30 to 53.268089, 40 fc rgb "#808080"
set object 48 rect from 0.094081, 30 to 53.816526, 40 fc rgb "#800080"
set object 49 rect from 0.095914, 30 to 55.209453, 40 fc rgb "#008080"
set object 50 rect from 0.097501, 30 to 144.996487, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.266106, 40 to 160.209217, 50 fc rgb "#FF0000"
set object 52 rect from 0.202795, 40 to 115.969059, 50 fc rgb "#00FF00"
set object 53 rect from 0.204915, 40 to 116.422781, 50 fc rgb "#0000FF"
set object 54 rect from 0.205392, 40 to 118.699331, 50 fc rgb "#FFFF00"
set object 55 rect from 0.209458, 40 to 121.034864, 50 fc rgb "#FF00FF"
set object 56 rect from 0.213523, 40 to 125.382655, 50 fc rgb "#808080"
set object 57 rect from 0.221190, 40 to 125.733722, 50 fc rgb "#800080"
set object 58 rect from 0.222066, 40 to 126.071745, 50 fc rgb "#008080"
set object 59 rect from 0.222409, 40 to 150.645326, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.262408, 50 to 158.623457, 60 fc rgb "#FF0000"
set object 61 rect from 0.139014, 50 to 79.968491, 60 fc rgb "#00FF00"
set object 62 rect from 0.141454, 50 to 80.536778, 60 fc rgb "#0000FF"
set object 63 rect from 0.142117, 50 to 82.556975, 60 fc rgb "#FFFF00"
set object 64 rect from 0.145717, 50 to 85.442079, 60 fc rgb "#FF00FF"
set object 65 rect from 0.150809, 50 to 89.871540, 60 fc rgb "#808080"
set object 66 rect from 0.158586, 50 to 90.271382, 60 fc rgb "#800080"
set object 67 rect from 0.159554, 50 to 90.688806, 60 fc rgb "#008080"
set object 68 rect from 0.160037, 50 to 148.409044, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
