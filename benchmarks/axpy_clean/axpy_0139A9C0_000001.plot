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

set object 15 rect from 0.186518, 0 to 153.616321, 10 fc rgb "#FF0000"
set object 16 rect from 0.086754, 0 to 106.216120, 10 fc rgb "#00FF00"
set object 17 rect from 0.132316, 0 to 106.957138, 10 fc rgb "#0000FF"
set object 18 rect from 0.132904, 0 to 107.676409, 10 fc rgb "#FFFF00"
set object 19 rect from 0.133850, 0 to 108.716251, 10 fc rgb "#FF00FF"
set object 20 rect from 0.135124, 0 to 109.835833, 10 fc rgb "#808080"
set object 21 rect from 0.136520, 0 to 110.339242, 10 fc rgb "#800080"
set object 22 rect from 0.137414, 0 to 110.847485, 10 fc rgb "#008080"
set object 23 rect from 0.137774, 0 to 149.744504, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.188181, 10 to 154.423392, 20 fc rgb "#FF0000"
set object 25 rect from 0.148879, 10 to 121.453712, 20 fc rgb "#00FF00"
set object 26 rect from 0.151162, 10 to 122.017531, 20 fc rgb "#0000FF"
set object 27 rect from 0.151600, 10 to 122.653840, 20 fc rgb "#FFFF00"
set object 28 rect from 0.152402, 10 to 123.508428, 20 fc rgb "#FF00FF"
set object 29 rect from 0.153462, 10 to 124.487055, 20 fc rgb "#808080"
set object 30 rect from 0.154681, 10 to 124.903475, 20 fc rgb "#800080"
set object 31 rect from 0.155463, 10 to 125.357751, 20 fc rgb "#008080"
set object 32 rect from 0.155760, 10 to 151.059811, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.184791, 20 to 151.930510, 30 fc rgb "#FF0000"
set object 34 rect from 0.070094, 20 to 57.911390, 30 fc rgb "#00FF00"
set object 35 rect from 0.072334, 20 to 58.471180, 30 fc rgb "#0000FF"
set object 36 rect from 0.072706, 20 to 59.203339, 30 fc rgb "#FFFF00"
set object 37 rect from 0.073617, 20 to 60.200492, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074854, 20 to 61.143679, 30 fc rgb "#808080"
set object 39 rect from 0.076042, 20 to 61.640644, 30 fc rgb "#800080"
set object 40 rect from 0.076932, 20 to 62.173050, 30 fc rgb "#008080"
set object 41 rect from 0.077323, 20 to 148.343012, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.189714, 30 to 155.951336, 40 fc rgb "#FF0000"
set object 43 rect from 0.165942, 30 to 135.108585, 40 fc rgb "#00FF00"
set object 44 rect from 0.168088, 30 to 135.634548, 40 fc rgb "#0000FF"
set object 45 rect from 0.168507, 30 to 136.325627, 40 fc rgb "#FFFF00"
set object 46 rect from 0.169383, 30 to 137.365471, 40 fc rgb "#FF00FF"
set object 47 rect from 0.170675, 30 to 138.322350, 40 fc rgb "#808080"
set object 48 rect from 0.171865, 30 to 138.849923, 40 fc rgb "#800080"
set object 49 rect from 0.172791, 30 to 139.429851, 40 fc rgb "#008080"
set object 50 rect from 0.173235, 30 to 152.390423, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.182894, 40 to 150.375175, 50 fc rgb "#FF0000"
set object 52 rect from 0.052568, 40 to 44.104283, 50 fc rgb "#00FF00"
set object 53 rect from 0.055130, 40 to 44.630246, 50 fc rgb "#0000FF"
set object 54 rect from 0.055539, 40 to 45.387374, 50 fc rgb "#FFFF00"
set object 55 rect from 0.056471, 40 to 46.318480, 50 fc rgb "#FF00FF"
set object 56 rect from 0.057636, 40 to 47.291469, 50 fc rgb "#808080"
set object 57 rect from 0.058827, 40 to 47.774741, 50 fc rgb "#800080"
set object 58 rect from 0.059678, 40 to 48.258820, 50 fc rgb "#008080"
set object 59 rect from 0.060059, 40 to 146.757073, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.180914, 50 to 151.686456, 60 fc rgb "#FF0000"
set object 61 rect from 0.029430, 50 to 26.054849, 60 fc rgb "#00FF00"
set object 62 rect from 0.032881, 50 to 27.675423, 60 fc rgb "#0000FF"
set object 63 rect from 0.034489, 50 to 29.350770, 60 fc rgb "#FFFF00"
set object 64 rect from 0.036604, 50 to 30.849722, 60 fc rgb "#FF00FF"
set object 65 rect from 0.038423, 50 to 32.161002, 60 fc rgb "#808080"
set object 66 rect from 0.040062, 50 to 32.789256, 60 fc rgb "#800080"
set object 67 rect from 0.041326, 50 to 33.831515, 60 fc rgb "#008080"
set object 68 rect from 0.042157, 50 to 144.943993, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
