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

set object 15 rect from 0.182473, 0 to 155.513247, 10 fc rgb "#FF0000"
set object 16 rect from 0.159943, 0 to 135.303629, 10 fc rgb "#00FF00"
set object 17 rect from 0.162213, 0 to 135.825211, 10 fc rgb "#0000FF"
set object 18 rect from 0.162607, 0 to 136.449604, 10 fc rgb "#FFFF00"
set object 19 rect from 0.163358, 0 to 137.351506, 10 fc rgb "#FF00FF"
set object 20 rect from 0.164446, 0 to 138.444821, 10 fc rgb "#808080"
set object 21 rect from 0.165757, 0 to 138.881980, 10 fc rgb "#800080"
set object 22 rect from 0.166515, 0 to 139.329170, 10 fc rgb "#008080"
set object 23 rect from 0.166802, 0 to 152.064455, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.179229, 10 to 153.174488, 20 fc rgb "#FF0000"
set object 25 rect from 0.129701, 10 to 109.992709, 20 fc rgb "#00FF00"
set object 26 rect from 0.131967, 10 to 110.698182, 20 fc rgb "#0000FF"
set object 27 rect from 0.132565, 10 to 111.377743, 20 fc rgb "#FFFF00"
set object 28 rect from 0.133401, 10 to 112.410040, 20 fc rgb "#FF00FF"
set object 29 rect from 0.134629, 10 to 113.570225, 20 fc rgb "#808080"
set object 30 rect from 0.136030, 10 to 114.065058, 20 fc rgb "#800080"
set object 31 rect from 0.136852, 10 to 114.551534, 20 fc rgb "#008080"
set object 32 rect from 0.137162, 10 to 149.296896, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.176000, 20 to 150.527293, 30 fc rgb "#FF0000"
set object 34 rect from 0.097844, 20 to 83.514918, 30 fc rgb "#00FF00"
set object 35 rect from 0.100284, 20 to 84.130117, 30 fc rgb "#0000FF"
set object 36 rect from 0.100764, 20 to 84.915833, 30 fc rgb "#FFFF00"
set object 37 rect from 0.101720, 20 to 86.037568, 30 fc rgb "#FF00FF"
set object 38 rect from 0.103061, 20 to 87.049804, 30 fc rgb "#808080"
set object 39 rect from 0.104297, 20 to 87.586431, 30 fc rgb "#800080"
set object 40 rect from 0.105165, 20 to 88.168195, 30 fc rgb "#008080"
set object 41 rect from 0.105641, 20 to 146.635492, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.174314, 30 to 151.641506, 40 fc rgb "#FF0000"
set object 43 rect from 0.075556, 30 to 65.720124, 40 fc rgb "#00FF00"
set object 44 rect from 0.079151, 30 to 67.014883, 40 fc rgb "#0000FF"
set object 45 rect from 0.080293, 30 to 68.886392, 40 fc rgb "#FFFF00"
set object 46 rect from 0.082595, 30 to 70.573173, 40 fc rgb "#FF00FF"
set object 47 rect from 0.084569, 30 to 71.830319, 40 fc rgb "#808080"
set object 48 rect from 0.086073, 30 to 72.474773, 40 fc rgb "#800080"
set object 49 rect from 0.087361, 30 to 73.588149, 40 fc rgb "#008080"
set object 50 rect from 0.088189, 30 to 144.865124, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.180863, 40 to 154.463396, 50 fc rgb "#FF0000"
set object 52 rect from 0.143850, 40 to 121.612947, 50 fc rgb "#00FF00"
set object 53 rect from 0.145855, 40 to 122.218951, 50 fc rgb "#0000FF"
set object 54 rect from 0.146342, 40 to 122.925260, 50 fc rgb "#FFFF00"
set object 55 rect from 0.147176, 40 to 124.045323, 50 fc rgb "#FF00FF"
set object 56 rect from 0.148530, 40 to 125.015766, 50 fc rgb "#808080"
set object 57 rect from 0.149701, 40 to 125.526481, 50 fc rgb "#800080"
set object 58 rect from 0.150562, 40 to 126.057257, 50 fc rgb "#008080"
set object 59 rect from 0.150929, 40 to 150.671063, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.177534, 50 to 151.575471, 60 fc rgb "#FF0000"
set object 61 rect from 0.113937, 50 to 96.713274, 60 fc rgb "#00FF00"
set object 62 rect from 0.116149, 50 to 97.422090, 60 fc rgb "#0000FF"
set object 63 rect from 0.116666, 50 to 98.154311, 60 fc rgb "#FFFF00"
set object 64 rect from 0.117559, 50 to 99.085468, 60 fc rgb "#FF00FF"
set object 65 rect from 0.118656, 50 to 100.040029, 60 fc rgb "#808080"
set object 66 rect from 0.119824, 50 to 100.590866, 60 fc rgb "#800080"
set object 67 rect from 0.120729, 50 to 101.109104, 60 fc rgb "#008080"
set object 68 rect from 0.121079, 50 to 147.947805, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0