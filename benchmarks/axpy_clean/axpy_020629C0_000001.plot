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

set object 15 rect from 0.148904, 0 to 155.786908, 10 fc rgb "#FF0000"
set object 16 rect from 0.097361, 0 to 101.477981, 10 fc rgb "#00FF00"
set object 17 rect from 0.099861, 0 to 102.235827, 10 fc rgb "#0000FF"
set object 18 rect from 0.100345, 0 to 103.029374, 10 fc rgb "#FFFF00"
set object 19 rect from 0.101152, 0 to 104.196232, 10 fc rgb "#FF00FF"
set object 20 rect from 0.102294, 0 to 105.570148, 10 fc rgb "#808080"
set object 21 rect from 0.103644, 0 to 106.192338, 10 fc rgb "#800080"
set object 22 rect from 0.104490, 0 to 106.781887, 10 fc rgb "#008080"
set object 23 rect from 0.104814, 0 to 151.205151, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.145288, 10 to 152.812643, 20 fc rgb "#FF0000"
set object 25 rect from 0.062686, 10 to 66.310032, 20 fc rgb "#00FF00"
set object 26 rect from 0.065375, 10 to 67.339194, 20 fc rgb "#0000FF"
set object 27 rect from 0.066150, 10 to 68.287777, 20 fc rgb "#FFFF00"
set object 28 rect from 0.067111, 10 to 69.615793, 20 fc rgb "#FF00FF"
set object 29 rect from 0.068420, 10 to 71.144745, 20 fc rgb "#808080"
set object 30 rect from 0.069897, 10 to 71.803654, 20 fc rgb "#800080"
set object 31 rect from 0.070824, 10 to 72.481941, 20 fc rgb "#008080"
set object 32 rect from 0.071188, 10 to 147.416938, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.143166, 20 to 154.314052, 30 fc rgb "#FF0000"
set object 34 rect from 0.038088, 20 to 42.048746, 30 fc rgb "#00FF00"
set object 35 rect from 0.041834, 20 to 43.794954, 30 fc rgb "#0000FF"
set object 36 rect from 0.043076, 20 to 46.577465, 30 fc rgb "#FFFF00"
set object 37 rect from 0.045887, 20 to 48.941783, 30 fc rgb "#FF00FF"
set object 38 rect from 0.048124, 20 to 50.650253, 30 fc rgb "#808080"
set object 39 rect from 0.049811, 20 to 51.490718, 30 fc rgb "#800080"
set object 40 rect from 0.051108, 20 to 52.742235, 30 fc rgb "#008080"
set object 41 rect from 0.051854, 20 to 145.040381, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.150470, 30 to 157.605536, 40 fc rgb "#FF0000"
set object 43 rect from 0.112862, 30 to 116.876645, 40 fc rgb "#00FF00"
set object 44 rect from 0.114962, 30 to 117.585533, 40 fc rgb "#0000FF"
set object 45 rect from 0.115398, 30 to 118.531056, 40 fc rgb "#FFFF00"
set object 46 rect from 0.116336, 30 to 119.851933, 40 fc rgb "#FF00FF"
set object 47 rect from 0.117633, 30 to 121.098350, 40 fc rgb "#808080"
set object 48 rect from 0.118855, 30 to 121.783777, 40 fc rgb "#800080"
set object 49 rect from 0.119784, 30 to 122.439625, 40 fc rgb "#008080"
set object 50 rect from 0.120163, 30 to 152.932999, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.152047, 40 to 158.942731, 50 fc rgb "#FF0000"
set object 52 rect from 0.128581, 40 to 132.902598, 50 fc rgb "#00FF00"
set object 53 rect from 0.130668, 40 to 133.640045, 50 fc rgb "#0000FF"
set object 54 rect from 0.131138, 40 to 134.532529, 50 fc rgb "#FFFF00"
set object 55 rect from 0.132012, 40 to 135.700408, 50 fc rgb "#FF00FF"
set object 56 rect from 0.133161, 40 to 136.901947, 50 fc rgb "#808080"
set object 57 rect from 0.134338, 40 to 137.501696, 50 fc rgb "#800080"
set object 58 rect from 0.135192, 40 to 138.111645, 50 fc rgb "#008080"
set object 59 rect from 0.135524, 40 to 154.591489, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.147095, 50 to 154.627189, 60 fc rgb "#FF0000"
set object 61 rect from 0.080047, 50 to 83.642580, 60 fc rgb "#00FF00"
set object 62 rect from 0.082366, 50 to 84.524866, 60 fc rgb "#0000FF"
set object 63 rect from 0.082983, 50 to 85.496908, 60 fc rgb "#FFFF00"
set object 64 rect from 0.083957, 50 to 86.997301, 60 fc rgb "#FF00FF"
set object 65 rect from 0.085442, 50 to 88.301858, 60 fc rgb "#808080"
set object 66 rect from 0.086726, 50 to 89.050524, 60 fc rgb "#800080"
set object 67 rect from 0.087704, 50 to 89.793071, 60 fc rgb "#008080"
set object 68 rect from 0.088160, 50 to 149.254946, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
