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

set object 15 rect from 0.126087, 0 to 160.711899, 10 fc rgb "#FF0000"
set object 16 rect from 0.105301, 0 to 133.052000, 10 fc rgb "#00FF00"
set object 17 rect from 0.107511, 0 to 133.816899, 10 fc rgb "#0000FF"
set object 18 rect from 0.107885, 0 to 134.707171, 10 fc rgb "#FFFF00"
set object 19 rect from 0.108602, 0 to 135.961278, 10 fc rgb "#FF00FF"
set object 20 rect from 0.109608, 0 to 137.458769, 10 fc rgb "#808080"
set object 21 rect from 0.110819, 0 to 138.073389, 10 fc rgb "#800080"
set object 22 rect from 0.111560, 0 to 138.715356, 10 fc rgb "#008080"
set object 23 rect from 0.111833, 0 to 155.899108, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.124538, 10 to 159.332380, 20 fc rgb "#FF0000"
set object 25 rect from 0.089793, 10 to 113.851759, 20 fc rgb "#00FF00"
set object 26 rect from 0.092055, 10 to 114.713500, 20 fc rgb "#0000FF"
set object 27 rect from 0.092494, 10 to 115.654691, 20 fc rgb "#FFFF00"
set object 28 rect from 0.093307, 10 to 117.160878, 20 fc rgb "#FF00FF"
set object 29 rect from 0.094485, 10 to 118.736561, 20 fc rgb "#808080"
set object 30 rect from 0.095757, 10 to 119.491505, 20 fc rgb "#800080"
set object 31 rect from 0.096653, 10 to 120.240269, 20 fc rgb "#008080"
set object 32 rect from 0.096954, 10 to 153.917367, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.122888, 20 to 157.061295, 30 fc rgb "#FF0000"
set object 34 rect from 0.075153, 20 to 95.489648, 30 fc rgb "#00FF00"
set object 35 rect from 0.077260, 20 to 96.289295, 30 fc rgb "#0000FF"
set object 36 rect from 0.077675, 20 to 97.345979, 30 fc rgb "#FFFF00"
set object 37 rect from 0.078510, 20 to 98.788813, 30 fc rgb "#FF00FF"
set object 38 rect from 0.079672, 20 to 100.195642, 30 fc rgb "#808080"
set object 39 rect from 0.080808, 20 to 100.845046, 30 fc rgb "#800080"
set object 40 rect from 0.081608, 20 to 101.648430, 30 fc rgb "#008080"
set object 41 rect from 0.081998, 20 to 151.992724, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.121319, 30 to 155.310577, 40 fc rgb "#FF0000"
set object 43 rect from 0.059664, 30 to 76.422255, 40 fc rgb "#00FF00"
set object 44 rect from 0.061919, 30 to 77.182158, 40 fc rgb "#0000FF"
set object 45 rect from 0.062270, 30 to 78.305896, 40 fc rgb "#FFFF00"
set object 46 rect from 0.063178, 30 to 79.863002, 40 fc rgb "#FF00FF"
set object 47 rect from 0.064431, 30 to 81.264835, 40 fc rgb "#808080"
set object 48 rect from 0.065560, 30 to 81.955242, 40 fc rgb "#800080"
set object 49 rect from 0.066387, 30 to 82.711407, 40 fc rgb "#008080"
set object 50 rect from 0.066731, 30 to 149.952663, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.119740, 40 to 153.554789, 50 fc rgb "#FF0000"
set object 52 rect from 0.044818, 40 to 58.137153, 50 fc rgb "#00FF00"
set object 53 rect from 0.047193, 40 to 59.006332, 50 fc rgb "#0000FF"
set object 54 rect from 0.047650, 40 to 60.100244, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048515, 40 to 61.677185, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049798, 40 to 63.203244, 50 fc rgb "#808080"
set object 57 rect from 0.051031, 40 to 63.927141, 50 fc rgb "#800080"
set object 58 rect from 0.051887, 40 to 64.776448, 50 fc rgb "#008080"
set object 59 rect from 0.052302, 40 to 147.989535, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.118044, 50 to 154.601594, 60 fc rgb "#FF0000"
set object 61 rect from 0.023763, 50 to 32.994166, 60 fc rgb "#00FF00"
set object 62 rect from 0.027051, 50 to 34.661770, 60 fc rgb "#0000FF"
set object 63 rect from 0.028045, 50 to 37.125283, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030109, 50 to 39.357809, 60 fc rgb "#FF00FF"
set object 65 rect from 0.031825, 50 to 41.251404, 60 fc rgb "#808080"
set object 66 rect from 0.033354, 50 to 42.213725, 60 fc rgb "#800080"
set object 67 rect from 0.034666, 50 to 43.953305, 60 fc rgb "#008080"
set object 68 rect from 0.035550, 50 to 145.358388, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
