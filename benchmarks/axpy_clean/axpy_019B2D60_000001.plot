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

set object 15 rect from 0.210088, 0 to 156.627141, 10 fc rgb "#FF0000"
set object 16 rect from 0.096758, 0 to 69.684242, 10 fc rgb "#00FF00"
set object 17 rect from 0.098845, 0 to 70.143494, 10 fc rgb "#0000FF"
set object 18 rect from 0.099243, 0 to 70.633189, 10 fc rgb "#FFFF00"
set object 19 rect from 0.099977, 0 to 71.403098, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101023, 0 to 77.807247, 10 fc rgb "#808080"
set object 21 rect from 0.110074, 0 to 78.192212, 10 fc rgb "#800080"
set object 22 rect from 0.110877, 0 to 78.568678, 10 fc rgb "#008080"
set object 23 rect from 0.111152, 0 to 148.162341, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.208008, 10 to 155.613102, 20 fc rgb "#FF0000"
set object 25 rect from 0.071904, 10 to 52.337126, 20 fc rgb "#00FF00"
set object 26 rect from 0.074337, 10 to 52.906781, 20 fc rgb "#0000FF"
set object 27 rect from 0.074883, 10 to 53.465111, 20 fc rgb "#FFFF00"
set object 28 rect from 0.075704, 10 to 54.395657, 20 fc rgb "#FF00FF"
set object 29 rect from 0.077022, 10 to 60.888266, 20 fc rgb "#808080"
set object 30 rect from 0.086213, 10 to 61.354593, 20 fc rgb "#800080"
set object 31 rect from 0.087117, 10 to 61.807487, 20 fc rgb "#008080"
set object 32 rect from 0.087486, 10 to 146.655771, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.211878, 20 to 162.362579, 30 fc rgb "#FF0000"
set object 34 rect from 0.120399, 20 to 86.347062, 30 fc rgb "#00FF00"
set object 35 rect from 0.122387, 20 to 86.804205, 30 fc rgb "#0000FF"
set object 36 rect from 0.122789, 20 to 88.789134, 30 fc rgb "#FFFF00"
set object 37 rect from 0.125611, 20 to 92.356355, 30 fc rgb "#FF00FF"
set object 38 rect from 0.130646, 20 to 98.807206, 30 fc rgb "#808080"
set object 39 rect from 0.139776, 20 to 99.333691, 30 fc rgb "#800080"
set object 40 rect from 0.140813, 20 to 99.862296, 30 fc rgb "#008080"
set object 41 rect from 0.141245, 20 to 149.455202, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.213448, 30 to 163.271171, 40 fc rgb "#FF0000"
set object 43 rect from 0.148475, 30 to 106.040009, 40 fc rgb "#00FF00"
set object 44 rect from 0.150215, 30 to 106.485817, 40 fc rgb "#0000FF"
set object 45 rect from 0.150601, 30 to 108.536566, 40 fc rgb "#FFFF00"
set object 46 rect from 0.153502, 30 to 111.997633, 40 fc rgb "#FF00FF"
set object 47 rect from 0.158402, 30 to 118.321116, 40 fc rgb "#808080"
set object 48 rect from 0.167342, 30 to 118.803724, 40 fc rgb "#800080"
set object 49 rect from 0.168296, 30 to 119.260868, 40 fc rgb "#008080"
set object 50 rect from 0.168654, 30 to 150.703481, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.214940, 40 to 164.700617, 50 fc rgb "#FF0000"
set object 52 rect from 0.177921, 40 to 126.877201, 50 fc rgb "#00FF00"
set object 53 rect from 0.179661, 40 to 127.337877, 50 fc rgb "#0000FF"
set object 54 rect from 0.180067, 40 to 129.368106, 50 fc rgb "#FFFF00"
set object 55 rect from 0.182936, 40 to 133.163893, 50 fc rgb "#FF00FF"
set object 56 rect from 0.188313, 40 to 139.526285, 50 fc rgb "#808080"
set object 57 rect from 0.197295, 40 to 140.001112, 50 fc rgb "#800080"
set object 58 rect from 0.198243, 40 to 140.435605, 50 fc rgb "#008080"
set object 59 rect from 0.198578, 40 to 151.753614, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.206110, 50 to 161.121365, 60 fc rgb "#FF0000"
set object 61 rect from 0.035147, 50 to 27.317769, 60 fc rgb "#00FF00"
set object 62 rect from 0.039160, 50 to 28.204450, 60 fc rgb "#0000FF"
set object 63 rect from 0.039991, 50 to 31.523280, 60 fc rgb "#FFFF00"
set object 64 rect from 0.044713, 50 to 36.055718, 60 fc rgb "#FF00FF"
set object 65 rect from 0.051122, 50 to 42.699052, 60 fc rgb "#808080"
set object 66 rect from 0.060490, 50 to 43.326734, 60 fc rgb "#800080"
set object 67 rect from 0.061826, 50 to 44.264356, 60 fc rgb "#008080"
set object 68 rect from 0.062680, 50 to 145.034570, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
