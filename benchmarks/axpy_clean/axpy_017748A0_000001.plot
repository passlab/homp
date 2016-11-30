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

set object 15 rect from 0.234961, 0 to 152.699574, 10 fc rgb "#FF0000"
set object 16 rect from 0.174700, 0 to 112.472827, 10 fc rgb "#00FF00"
set object 17 rect from 0.176784, 0 to 112.926739, 10 fc rgb "#0000FF"
set object 18 rect from 0.177255, 0 to 113.406154, 10 fc rgb "#FFFF00"
set object 19 rect from 0.178028, 0 to 114.110613, 10 fc rgb "#FF00FF"
set object 20 rect from 0.179142, 0 to 115.507415, 10 fc rgb "#808080"
set object 21 rect from 0.181322, 0 to 115.884821, 10 fc rgb "#800080"
set object 22 rect from 0.182155, 0 to 116.244382, 10 fc rgb "#008080"
set object 23 rect from 0.182463, 0 to 149.381931, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.228971, 10 to 151.672522, 20 fc rgb "#FF0000"
set object 25 rect from 0.051743, 10 to 35.207565, 20 fc rgb "#00FF00"
set object 26 rect from 0.055807, 10 to 36.484513, 20 fc rgb "#0000FF"
set object 27 rect from 0.057364, 10 to 37.893426, 20 fc rgb "#FFFF00"
set object 28 rect from 0.059597, 10 to 39.020558, 20 fc rgb "#FF00FF"
set object 29 rect from 0.061346, 10 to 41.139033, 20 fc rgb "#808080"
set object 30 rect from 0.064674, 10 to 41.558518, 20 fc rgb "#800080"
set object 31 rect from 0.065758, 10 to 42.078735, 20 fc rgb "#008080"
set object 32 rect from 0.066158, 10 to 145.141795, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.236556, 20 to 154.734531, 30 fc rgb "#FF0000"
set object 34 rect from 0.191882, 20 to 123.313822, 30 fc rgb "#00FF00"
set object 35 rect from 0.193795, 20 to 123.769007, 30 fc rgb "#0000FF"
set object 36 rect from 0.194263, 20 to 125.314989, 30 fc rgb "#FFFF00"
set object 37 rect from 0.196733, 20 to 126.357967, 30 fc rgb "#FF00FF"
set object 38 rect from 0.198341, 20 to 127.346759, 30 fc rgb "#808080"
set object 39 rect from 0.199892, 20 to 127.755410, 30 fc rgb "#800080"
set object 40 rect from 0.200776, 20 to 128.181270, 30 fc rgb "#008080"
set object 41 rect from 0.201186, 20 to 150.433834, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.238000, 30 to 155.094724, 40 fc rgb "#FF0000"
set object 43 rect from 0.211459, 30 to 135.693782, 40 fc rgb "#00FF00"
set object 44 rect from 0.213239, 30 to 136.087129, 40 fc rgb "#0000FF"
set object 45 rect from 0.213587, 30 to 137.365355, 40 fc rgb "#FFFF00"
set object 46 rect from 0.215591, 30 to 138.178824, 40 fc rgb "#FF00FF"
set object 47 rect from 0.216867, 30 to 139.147215, 40 fc rgb "#808080"
set object 48 rect from 0.218400, 30 to 139.554588, 40 fc rgb "#800080"
set object 49 rect from 0.219341, 30 to 140.000213, 40 fc rgb "#008080"
set object 50 rect from 0.219732, 30 to 151.446214, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.233161, 40 to 152.629447, 50 fc rgb "#FF0000"
set object 52 rect from 0.154567, 40 to 99.583489, 50 fc rgb "#00FF00"
set object 53 rect from 0.156577, 40 to 100.078203, 50 fc rgb "#0000FF"
set object 54 rect from 0.157100, 40 to 101.567447, 50 fc rgb "#FFFF00"
set object 55 rect from 0.159436, 40 to 102.621265, 50 fc rgb "#FF00FF"
set object 56 rect from 0.161106, 40 to 103.606864, 50 fc rgb "#808080"
set object 57 rect from 0.162650, 40 to 104.054403, 50 fc rgb "#800080"
set object 58 rect from 0.163647, 40 to 104.537005, 50 fc rgb "#008080"
set object 59 rect from 0.164098, 40 to 148.178934, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.231309, 50 to 152.468792, 60 fc rgb "#FF0000"
set object 61 rect from 0.130065, 50 to 84.383787, 60 fc rgb "#00FF00"
set object 62 rect from 0.132723, 50 to 85.143710, 60 fc rgb "#0000FF"
set object 63 rect from 0.133677, 50 to 86.872657, 60 fc rgb "#FFFF00"
set object 64 rect from 0.136414, 50 to 88.223555, 60 fc rgb "#FF00FF"
set object 65 rect from 0.138545, 50 to 89.425916, 60 fc rgb "#808080"
set object 66 rect from 0.140424, 50 to 89.912342, 60 fc rgb "#800080"
set object 67 rect from 0.141530, 50 to 90.624448, 60 fc rgb "#008080"
set object 68 rect from 0.142311, 50 to 146.875847, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
