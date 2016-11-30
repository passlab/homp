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

set object 15 rect from 0.248481, 0 to 156.454894, 10 fc rgb "#FF0000"
set object 16 rect from 0.158589, 0 to 96.419626, 10 fc rgb "#00FF00"
set object 17 rect from 0.161030, 0 to 96.844053, 10 fc rgb "#0000FF"
set object 18 rect from 0.161435, 0 to 97.335118, 10 fc rgb "#FFFF00"
set object 19 rect from 0.162307, 0 to 97.997273, 10 fc rgb "#FF00FF"
set object 20 rect from 0.163377, 0 to 103.864227, 10 fc rgb "#808080"
set object 21 rect from 0.173162, 0 to 104.228624, 10 fc rgb "#800080"
set object 22 rect from 0.173997, 0 to 104.556399, 10 fc rgb "#008080"
set object 23 rect from 0.174280, 0 to 148.616482, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.244420, 10 to 155.463161, 20 fc rgb "#FF0000"
set object 25 rect from 0.098732, 10 to 60.711833, 20 fc rgb "#00FF00"
set object 26 rect from 0.101528, 10 to 62.102781, 20 fc rgb "#0000FF"
set object 27 rect from 0.103564, 10 to 62.673687, 20 fc rgb "#FFFF00"
set object 28 rect from 0.104544, 10 to 63.524344, 20 fc rgb "#FF00FF"
set object 29 rect from 0.105985, 10 to 69.572596, 20 fc rgb "#808080"
set object 30 rect from 0.116037, 10 to 69.958003, 20 fc rgb "#800080"
set object 31 rect from 0.116965, 10 to 70.340409, 20 fc rgb "#008080"
set object 32 rect from 0.117323, 10 to 146.247008, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.242332, 20 to 168.126369, 30 fc rgb "#FF0000"
set object 34 rect from 0.039787, 20 to 26.097828, 30 fc rgb "#00FF00"
set object 35 rect from 0.044020, 20 to 29.020799, 30 fc rgb "#0000FF"
set object 36 rect from 0.048470, 20 to 31.680829, 30 fc rgb "#FFFF00"
set object 37 rect from 0.052939, 20 to 34.685443, 30 fc rgb "#FF00FF"
set object 38 rect from 0.057913, 20 to 48.567305, 30 fc rgb "#808080"
set object 39 rect from 0.081203, 20 to 49.420964, 30 fc rgb "#800080"
set object 40 rect from 0.082924, 20 to 50.201984, 30 fc rgb "#008080"
set object 41 rect from 0.083808, 20 to 144.757008, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.250176, 30 to 160.670962, 40 fc rgb "#FF0000"
set object 43 rect from 0.181979, 30 to 110.342910, 40 fc rgb "#00FF00"
set object 44 rect from 0.184166, 30 to 110.776343, 40 fc rgb "#0000FF"
set object 45 rect from 0.184642, 30 to 112.695574, 40 fc rgb "#FFFF00"
set object 46 rect from 0.187858, 30 to 115.018222, 40 fc rgb "#FF00FF"
set object 47 rect from 0.191744, 30 to 120.848557, 40 fc rgb "#808080"
set object 48 rect from 0.201434, 30 to 121.290394, 40 fc rgb "#800080"
set object 49 rect from 0.202430, 30 to 121.689008, 40 fc rgb "#008080"
set object 50 rect from 0.202822, 30 to 149.809923, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.251671, 40 to 161.210652, 50 fc rgb "#FF0000"
set object 52 rect from 0.212902, 40 to 128.891079, 50 fc rgb "#00FF00"
set object 53 rect from 0.215078, 40 to 129.250072, 50 fc rgb "#0000FF"
set object 54 rect from 0.215416, 40 to 131.123679, 50 fc rgb "#FFFF00"
set object 55 rect from 0.218547, 40 to 133.277637, 50 fc rgb "#FF00FF"
set object 56 rect from 0.222139, 40 to 139.060546, 50 fc rgb "#808080"
set object 57 rect from 0.231769, 40 to 139.471166, 50 fc rgb "#800080"
set object 58 rect from 0.232734, 40 to 139.858975, 50 fc rgb "#008080"
set object 59 rect from 0.233092, 40 to 150.788450, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.246434, 50 to 158.661680, 60 fc rgb "#FF0000"
set object 61 rect from 0.127651, 50 to 77.872657, 60 fc rgb "#00FF00"
set object 62 rect from 0.130184, 50 to 78.400340, 60 fc rgb "#0000FF"
set object 63 rect from 0.130729, 50 to 80.458847, 60 fc rgb "#FFFF00"
set object 64 rect from 0.134158, 50 to 82.816314, 60 fc rgb "#FF00FF"
set object 65 rect from 0.138107, 50 to 88.703679, 60 fc rgb "#808080"
set object 66 rect from 0.147910, 50 to 89.147316, 60 fc rgb "#800080"
set object 67 rect from 0.148881, 50 to 89.581350, 60 fc rgb "#008080"
set object 68 rect from 0.149357, 50 to 147.460260, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
