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

set object 15 rect from 0.208763, 0 to 155.584701, 10 fc rgb "#FF0000"
set object 16 rect from 0.041108, 0 to 31.026441, 10 fc rgb "#00FF00"
set object 17 rect from 0.045019, 0 to 32.022268, 10 fc rgb "#0000FF"
set object 18 rect from 0.045987, 0 to 33.409872, 10 fc rgb "#FFFF00"
set object 19 rect from 0.048008, 0 to 34.490195, 10 fc rgb "#FF00FF"
set object 20 rect from 0.049522, 0 to 41.099956, 10 fc rgb "#808080"
set object 21 rect from 0.058996, 0 to 41.548290, 10 fc rgb "#800080"
set object 22 rect from 0.060086, 0 to 42.144670, 10 fc rgb "#008080"
set object 23 rect from 0.060491, 0 to 144.813555, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.215994, 10 to 158.665794, 20 fc rgb "#FF0000"
set object 25 rect from 0.157629, 10 to 111.179527, 20 fc rgb "#00FF00"
set object 26 rect from 0.159575, 10 to 111.646019, 20 fc rgb "#0000FF"
set object 27 rect from 0.159995, 10 to 112.115997, 20 fc rgb "#FFFF00"
set object 28 rect from 0.160688, 10 to 112.886263, 20 fc rgb "#FF00FF"
set object 29 rect from 0.161798, 10 to 119.165018, 20 fc rgb "#808080"
set object 30 rect from 0.170764, 10 to 119.554693, 20 fc rgb "#800080"
set object 31 rect from 0.171599, 10 to 119.956230, 20 fc rgb "#008080"
set object 32 rect from 0.171898, 10 to 150.383502, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.210870, 20 to 160.706325, 30 fc rgb "#FF0000"
set object 34 rect from 0.069382, 20 to 49.887139, 30 fc rgb "#00FF00"
set object 35 rect from 0.071837, 20 to 50.410896, 30 fc rgb "#0000FF"
set object 36 rect from 0.072312, 20 to 53.002426, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076074, 20 to 57.054184, 30 fc rgb "#FF00FF"
set object 38 rect from 0.081857, 20 to 63.382514, 30 fc rgb "#808080"
set object 39 rect from 0.090920, 20 to 63.958644, 30 fc rgb "#800080"
set object 40 rect from 0.092055, 20 to 64.726121, 30 fc rgb "#008080"
set object 41 rect from 0.092838, 20 to 146.727007, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.217586, 30 to 164.564635, 40 fc rgb "#FF0000"
set object 43 rect from 0.180025, 30 to 126.770615, 40 fc rgb "#00FF00"
set object 44 rect from 0.181910, 30 to 127.473138, 40 fc rgb "#0000FF"
set object 45 rect from 0.182657, 30 to 129.539517, 40 fc rgb "#FFFF00"
set object 46 rect from 0.185649, 30 to 133.168780, 40 fc rgb "#FF00FF"
set object 47 rect from 0.190828, 30 to 139.441946, 40 fc rgb "#808080"
set object 48 rect from 0.199845, 30 to 139.965006, 40 fc rgb "#800080"
set object 49 rect from 0.200853, 30 to 140.472696, 40 fc rgb "#008080"
set object 50 rect from 0.201289, 30 to 151.545534, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.214314, 40 to 161.730071, 50 fc rgb "#FF0000"
set object 52 rect from 0.130709, 40 to 92.364917, 50 fc rgb "#00FF00"
set object 53 rect from 0.132629, 40 to 92.850265, 50 fc rgb "#0000FF"
set object 54 rect from 0.133089, 40 to 94.822366, 50 fc rgb "#FFFF00"
set object 55 rect from 0.135908, 40 to 98.242124, 50 fc rgb "#FF00FF"
set object 56 rect from 0.140800, 40 to 104.476184, 50 fc rgb "#808080"
set object 57 rect from 0.149748, 40 to 104.963624, 50 fc rgb "#800080"
set object 58 rect from 0.150704, 40 to 105.480399, 50 fc rgb "#008080"
set object 59 rect from 0.151184, 40 to 149.292699, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.212718, 50 to 160.996841, 60 fc rgb "#FF0000"
set object 61 rect from 0.100292, 50 to 71.168282, 60 fc rgb "#00FF00"
set object 62 rect from 0.102272, 50 to 71.636866, 60 fc rgb "#0000FF"
set object 63 rect from 0.102705, 50 to 73.796827, 60 fc rgb "#FFFF00"
set object 64 rect from 0.105795, 50 to 77.432364, 60 fc rgb "#FF00FF"
set object 65 rect from 0.111001, 50 to 83.650368, 60 fc rgb "#808080"
set object 66 rect from 0.119913, 50 to 84.126632, 60 fc rgb "#800080"
set object 67 rect from 0.120902, 50 to 84.684604, 60 fc rgb "#008080"
set object 68 rect from 0.121394, 50 to 148.025212, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
