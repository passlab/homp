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

set object 15 rect from 0.148300, 0 to 150.155173, 10 fc rgb "#FF0000"
set object 16 rect from 0.085545, 0 to 86.515938, 10 fc rgb "#00FF00"
set object 17 rect from 0.087479, 0 to 87.092645, 10 fc rgb "#0000FF"
set object 18 rect from 0.087843, 0 to 87.689201, 10 fc rgb "#FFFF00"
set object 19 rect from 0.088443, 0 to 88.594462, 10 fc rgb "#FF00FF"
set object 20 rect from 0.089370, 0 to 89.660526, 10 fc rgb "#808080"
set object 21 rect from 0.090451, 0 to 90.123083, 10 fc rgb "#800080"
set object 22 rect from 0.091152, 0 to 90.593579, 10 fc rgb "#008080"
set object 23 rect from 0.091385, 0 to 146.666150, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.146812, 10 to 150.326905, 20 fc rgb "#FF0000"
set object 25 rect from 0.068039, 10 to 69.990956, 20 fc rgb "#00FF00"
set object 26 rect from 0.070949, 10 to 71.029228, 20 fc rgb "#0000FF"
set object 27 rect from 0.071668, 10 to 72.322598, 20 fc rgb "#FFFF00"
set object 28 rect from 0.073020, 10 to 73.650713, 20 fc rgb "#FF00FF"
set object 29 rect from 0.074308, 10 to 74.836881, 20 fc rgb "#808080"
set object 30 rect from 0.075521, 10 to 75.415573, 20 fc rgb "#800080"
set object 31 rect from 0.076562, 10 to 76.211646, 20 fc rgb "#008080"
set object 32 rect from 0.076886, 10 to 144.923128, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.152344, 20 to 154.016429, 30 fc rgb "#FF0000"
set object 34 rect from 0.123801, 20 to 124.008824, 30 fc rgb "#00FF00"
set object 35 rect from 0.125235, 20 to 124.541853, 30 fc rgb "#0000FF"
set object 36 rect from 0.125570, 20 to 125.192015, 30 fc rgb "#FFFF00"
set object 37 rect from 0.126248, 20 to 126.246166, 30 fc rgb "#FF00FF"
set object 38 rect from 0.127283, 20 to 126.986654, 30 fc rgb "#808080"
set object 39 rect from 0.128039, 20 to 127.434320, 30 fc rgb "#800080"
set object 40 rect from 0.128702, 20 to 127.926655, 30 fc rgb "#008080"
set object 41 rect from 0.128983, 20 to 150.844048, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.149879, 30 to 152.116581, 40 fc rgb "#FF0000"
set object 43 rect from 0.098077, 30 to 98.840188, 40 fc rgb "#00FF00"
set object 44 rect from 0.099879, 30 to 99.436749, 40 fc rgb "#0000FF"
set object 45 rect from 0.100277, 30 to 100.209991, 40 fc rgb "#FFFF00"
set object 46 rect from 0.101085, 30 to 101.471601, 40 fc rgb "#FF00FF"
set object 47 rect from 0.102347, 30 to 102.256755, 40 fc rgb "#808080"
set object 48 rect from 0.103135, 30 to 102.851329, 40 fc rgb "#800080"
set object 49 rect from 0.103950, 30 to 103.710929, 40 fc rgb "#008080"
set object 50 rect from 0.104582, 30 to 148.125287, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.151186, 40 to 153.255100, 50 fc rgb "#FF0000"
set object 52 rect from 0.111897, 40 to 112.333738, 50 fc rgb "#00FF00"
set object 53 rect from 0.113473, 40 to 112.937246, 50 fc rgb "#0000FF"
set object 54 rect from 0.113878, 40 to 113.733319, 50 fc rgb "#FFFF00"
set object 55 rect from 0.114683, 40 to 114.635602, 50 fc rgb "#FF00FF"
set object 56 rect from 0.115592, 40 to 115.649056, 50 fc rgb "#808080"
set object 57 rect from 0.116619, 40 to 116.134442, 50 fc rgb "#800080"
set object 58 rect from 0.117321, 40 to 116.738941, 50 fc rgb "#008080"
set object 59 rect from 0.117710, 40 to 149.572513, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.153479, 50 to 154.999116, 60 fc rgb "#FF0000"
set object 61 rect from 0.135783, 50 to 135.904267, 60 fc rgb "#00FF00"
set object 62 rect from 0.137221, 50 to 136.395611, 60 fc rgb "#0000FF"
set object 63 rect from 0.137511, 50 to 137.039813, 60 fc rgb "#FFFF00"
set object 64 rect from 0.138160, 50 to 137.932171, 60 fc rgb "#FF00FF"
set object 65 rect from 0.139056, 50 to 138.673651, 60 fc rgb "#808080"
set object 66 rect from 0.139815, 50 to 139.166980, 60 fc rgb "#800080"
set object 67 rect from 0.140522, 50 to 139.711923, 60 fc rgb "#008080"
set object 68 rect from 0.140852, 50 to 152.000442, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
