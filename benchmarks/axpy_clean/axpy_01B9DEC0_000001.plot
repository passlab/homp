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

set object 15 rect from 0.337214, 0 to 154.510106, 10 fc rgb "#FF0000"
set object 16 rect from 0.085440, 0 to 38.154407, 10 fc rgb "#00FF00"
set object 17 rect from 0.089202, 0 to 38.723875, 10 fc rgb "#0000FF"
set object 18 rect from 0.090156, 0 to 39.524301, 10 fc rgb "#FFFF00"
set object 19 rect from 0.092050, 0 to 40.216343, 10 fc rgb "#FF00FF"
set object 20 rect from 0.093625, 0 to 47.721292, 10 fc rgb "#808080"
set object 21 rect from 0.111103, 0 to 48.024083, 10 fc rgb "#800080"
set object 22 rect from 0.112239, 0 to 48.377635, 10 fc rgb "#008080"
set object 23 rect from 0.112620, 0 to 144.505806, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.345484, 10 to 156.682157, 20 fc rgb "#FF0000"
set object 25 rect from 0.308096, 10 to 133.135483, 20 fc rgb "#00FF00"
set object 26 rect from 0.309876, 10 to 133.417202, 20 fc rgb "#0000FF"
set object 27 rect from 0.310307, 10 to 133.731613, 20 fc rgb "#FFFF00"
set object 28 rect from 0.311058, 10 to 134.153114, 20 fc rgb "#FF00FF"
set object 29 rect from 0.312041, 10 to 141.304518, 20 fc rgb "#808080"
set object 30 rect from 0.328658, 10 to 141.538065, 20 fc rgb "#800080"
set object 31 rect from 0.329462, 10 to 141.766883, 20 fc rgb "#008080"
set object 32 rect from 0.329729, 10 to 148.377634, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.342535, 20 to 160.939358, 30 fc rgb "#FF0000"
set object 34 rect from 0.223764, 20 to 96.847740, 30 fc rgb "#00FF00"
set object 35 rect from 0.225506, 20 to 97.097631, 30 fc rgb "#0000FF"
set object 36 rect from 0.225860, 20 to 98.294191, 30 fc rgb "#FFFF00"
set object 37 rect from 0.228654, 20 to 103.282581, 30 fc rgb "#FF00FF"
set object 38 rect from 0.240256, 20 to 110.314839, 30 fc rgb "#808080"
set object 39 rect from 0.256604, 20 to 110.762150, 30 fc rgb "#800080"
set object 40 rect from 0.257916, 20 to 111.051181, 30 fc rgb "#008080"
set object 41 rect from 0.258315, 20 to 147.103652, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.339308, 30 to 160.575058, 40 fc rgb "#FF0000"
set object 43 rect from 0.126558, 30 to 55.439138, 40 fc rgb "#00FF00"
set object 44 rect from 0.129251, 30 to 55.845161, 40 fc rgb "#0000FF"
set object 45 rect from 0.129953, 30 to 57.348820, 40 fc rgb "#FFFF00"
set object 46 rect from 0.133492, 30 to 62.859352, 40 fc rgb "#FF00FF"
set object 47 rect from 0.146285, 30 to 69.932905, 40 fc rgb "#808080"
set object 48 rect from 0.162728, 30 to 70.411184, 40 fc rgb "#800080"
set object 49 rect from 0.164180, 30 to 70.934192, 40 fc rgb "#008080"
set object 50 rect from 0.165048, 30 to 145.616773, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.341107, 40 to 163.078703, 50 fc rgb "#FF0000"
set object 52 rect from 0.174020, 40 to 75.496774, 50 fc rgb "#00FF00"
set object 53 rect from 0.175868, 40 to 75.777633, 50 fc rgb "#0000FF"
set object 54 rect from 0.176290, 40 to 77.109675, 50 fc rgb "#FFFF00"
set object 55 rect from 0.179399, 40 to 84.683439, 50 fc rgb "#FF00FF"
set object 56 rect from 0.197006, 40 to 91.707096, 50 fc rgb "#808080"
set object 57 rect from 0.213355, 40 to 92.168603, 50 fc rgb "#800080"
set object 58 rect from 0.214655, 40 to 92.444733, 50 fc rgb "#008080"
set object 59 rect from 0.215058, 40 to 146.400856, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.344105, 50 to 161.889026, 60 fc rgb "#FF0000"
set object 61 rect from 0.265840, 50 to 114.890322, 60 fc rgb "#00FF00"
set object 62 rect from 0.267468, 50 to 115.189679, 60 fc rgb "#0000FF"
set object 63 rect from 0.267929, 50 to 116.376773, 60 fc rgb "#FFFF00"
set object 64 rect from 0.270685, 50 to 121.621507, 60 fc rgb "#FF00FF"
set object 65 rect from 0.282884, 50 to 128.657201, 60 fc rgb "#808080"
set object 66 rect from 0.299249, 50 to 129.078279, 60 fc rgb "#800080"
set object 67 rect from 0.300485, 50 to 129.353550, 60 fc rgb "#008080"
set object 68 rect from 0.300882, 50 to 147.701935, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
