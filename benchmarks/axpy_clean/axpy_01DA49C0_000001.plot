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

set object 15 rect from 0.132749, 0 to 160.150003, 10 fc rgb "#FF0000"
set object 16 rect from 0.111075, 0 to 132.923266, 10 fc rgb "#00FF00"
set object 17 rect from 0.113376, 0 to 133.665327, 10 fc rgb "#0000FF"
set object 18 rect from 0.113773, 0 to 134.533209, 10 fc rgb "#FFFF00"
set object 19 rect from 0.114516, 0 to 135.703305, 10 fc rgb "#FF00FF"
set object 20 rect from 0.115507, 0 to 137.179191, 10 fc rgb "#808080"
set object 21 rect from 0.116769, 0 to 137.774263, 10 fc rgb "#800080"
set object 22 rect from 0.117524, 0 to 138.419874, 10 fc rgb "#008080"
set object 23 rect from 0.117820, 0 to 155.431871, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.131089, 10 to 158.610611, 20 fc rgb "#FF0000"
set object 25 rect from 0.095547, 10 to 114.541191, 20 fc rgb "#00FF00"
set object 26 rect from 0.097745, 10 to 115.412612, 20 fc rgb "#0000FF"
set object 27 rect from 0.098253, 10 to 116.278146, 20 fc rgb "#FFFF00"
set object 28 rect from 0.099023, 10 to 117.689369, 20 fc rgb "#FF00FF"
set object 29 rect from 0.100205, 10 to 119.184041, 20 fc rgb "#808080"
set object 30 rect from 0.101482, 10 to 119.881417, 20 fc rgb "#800080"
set object 31 rect from 0.102326, 10 to 120.575253, 20 fc rgb "#008080"
set object 32 rect from 0.102664, 10 to 153.486768, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129436, 20 to 156.510283, 30 fc rgb "#FF0000"
set object 34 rect from 0.080749, 20 to 97.155236, 30 fc rgb "#00FF00"
set object 35 rect from 0.082982, 20 to 97.924319, 30 fc rgb "#0000FF"
set object 36 rect from 0.083382, 20 to 98.952158, 30 fc rgb "#FFFF00"
set object 37 rect from 0.084258, 20 to 100.293952, 30 fc rgb "#FF00FF"
set object 38 rect from 0.085396, 20 to 101.635782, 30 fc rgb "#808080"
set object 39 rect from 0.086543, 20 to 102.287280, 30 fc rgb "#800080"
set object 40 rect from 0.087347, 20 to 103.022297, 30 fc rgb "#008080"
set object 41 rect from 0.087747, 20 to 151.642813, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127870, 30 to 154.785104, 40 fc rgb "#FF0000"
set object 43 rect from 0.064133, 30 to 77.639514, 40 fc rgb "#00FF00"
set object 44 rect from 0.066374, 30 to 78.382732, 40 fc rgb "#0000FF"
set object 45 rect from 0.066762, 30 to 79.441132, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067666, 30 to 80.737084, 40 fc rgb "#FF00FF"
set object 47 rect from 0.068769, 30 to 82.114172, 40 fc rgb "#808080"
set object 48 rect from 0.069963, 30 to 82.839796, 40 fc rgb "#800080"
set object 49 rect from 0.070808, 30 to 83.711182, 40 fc rgb "#008080"
set object 50 rect from 0.071308, 30 to 149.742396, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126316, 40 to 153.156340, 50 fc rgb "#FF0000"
set object 52 rect from 0.047594, 40 to 58.447176, 50 fc rgb "#00FF00"
set object 53 rect from 0.050090, 40 to 59.259823, 50 fc rgb "#0000FF"
set object 54 rect from 0.050502, 40 to 60.287626, 50 fc rgb "#FFFF00"
set object 55 rect from 0.051380, 40 to 61.757624, 50 fc rgb "#FF00FF"
set object 56 rect from 0.052643, 40 to 63.179432, 50 fc rgb "#808080"
set object 57 rect from 0.053860, 40 to 63.949707, 50 fc rgb "#800080"
set object 58 rect from 0.054751, 40 to 64.931668, 50 fc rgb "#008080"
set object 59 rect from 0.055371, 40 to 147.822002, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.124465, 50 to 154.199425, 60 fc rgb "#FF0000"
set object 61 rect from 0.025038, 50 to 32.811560, 60 fc rgb "#00FF00"
set object 62 rect from 0.028359, 50 to 34.136918, 60 fc rgb "#0000FF"
set object 63 rect from 0.029170, 50 to 36.725317, 60 fc rgb "#FFFF00"
set object 64 rect from 0.031384, 50 to 38.985567, 60 fc rgb "#FF00FF"
set object 65 rect from 0.033298, 50 to 40.798961, 60 fc rgb "#808080"
set object 66 rect from 0.034842, 50 to 41.720956, 60 fc rgb "#800080"
set object 67 rect from 0.036136, 50 to 43.831868, 60 fc rgb "#008080"
set object 68 rect from 0.037417, 50 to 145.264199, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
