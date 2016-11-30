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

set object 15 rect from 16.726136, 0 to 149.404481, 10 fc rgb "#FF0000"
set object 16 rect from 1.883661, 0 to 16.238330, 10 fc rgb "#00FF00"
set object 17 rect from 1.885430, 0 to 16.243386, 10 fc rgb "#0000FF"
set object 18 rect from 1.885784, 0 to 16.248305, 10 fc rgb "#FFFF00"
set object 19 rect from 1.886371, 0 to 16.255877, 10 fc rgb "#FF00FF"
set object 20 rect from 1.887235, 0 to 21.563303, 10 fc rgb "#808080"
set object 21 rect from 2.503516, 0 to 21.569798, 10 fc rgb "#800080"
set object 22 rect from 2.504482, 0 to 21.575432, 10 fc rgb "#008080"
set object 23 rect from 2.504782, 0 to 144.072634, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 16.723420, 10 to 149.261445, 20 fc rgb "#FF0000"
set object 25 rect from 1.263176, 10 to 10.899307, 20 fc rgb "#00FF00"
set object 26 rect from 1.265658, 10 to 10.906499, 20 fc rgb "#0000FF"
set object 27 rect from 1.266229, 10 to 10.912779, 20 fc rgb "#FFFF00"
set object 28 rect from 1.266988, 10 to 10.922298, 20 fc rgb "#FF00FF"
set object 29 rect from 1.268086, 10 to 16.105148, 20 fc rgb "#808080"
set object 30 rect from 1.869830, 10 to 16.111316, 20 fc rgb "#800080"
set object 31 rect from 1.870848, 10 to 16.117242, 20 fc rgb "#008080"
set object 32 rect from 1.871181, 10 to 144.049091, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 16.731970, 20 to 167.950205, 30 fc rgb "#FF0000"
set object 34 rect from 8.513317, 20 to 73.347143, 30 fc rgb "#00FF00"
set object 35 rect from 8.515127, 20 to 73.353776, 30 fc rgb "#0000FF"
set object 36 rect from 8.515695, 20 to 83.850039, 30 fc rgb "#FFFF00"
set object 37 rect from 9.734244, 20 to 91.498817, 30 fc rgb "#FF00FF"
set object 38 rect from 10.622171, 20 to 96.774682, 30 fc rgb "#808080"
set object 39 rect from 11.234728, 20 to 97.174072, 30 fc rgb "#800080"
set object 40 rect from 11.281466, 20 to 97.184263, 30 fc rgb "#008080"
set object 41 rect from 11.282244, 20 to 144.124473, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 16.720629, 30 to 154.063666, 40 fc rgb "#FF0000"
set object 43 rect from 0.033899, 30 to 0.316429, 40 fc rgb "#00FF00"
set object 44 rect from 0.037217, 30 to 0.325947, 40 fc rgb "#0000FF"
set object 45 rect from 0.037952, 30 to 0.387899, 40 fc rgb "#FFFF00"
set object 46 rect from 0.045170, 30 to 4.619267, 40 fc rgb "#FF00FF"
set object 47 rect from 0.536375, 30 to 9.944852, 40 fc rgb "#808080"
set object 48 rect from 1.154652, 30 to 10.355588, 40 fc rgb "#800080"
set object 49 rect from 1.203002, 30 to 10.765660, 40 fc rgb "#008080"
set object 50 rect from 1.249938, 30 to 144.023430, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 16.730148, 40 to 154.591432, 50 fc rgb "#FF0000"
set object 52 rect from 8.506162, 40 to 73.295209, 50 fc rgb "#00FF00"
set object 53 rect from 8.509252, 40 to 73.305528, 50 fc rgb "#0000FF"
set object 54 rect from 8.510107, 40 to 73.486923, 50 fc rgb "#FFFF00"
set object 55 rect from 8.531195, 40 to 78.243929, 50 fc rgb "#FF00FF"
set object 56 rect from 9.083418, 40 to 83.424781, 50 fc rgb "#808080"
set object 57 rect from 9.684941, 40 to 83.780421, 50 fc rgb "#800080"
set object 58 rect from 9.726546, 40 to 83.794367, 50 fc rgb "#008080"
set object 59 rect from 9.727755, 40 to 144.108012, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 16.727742, 50 to 266.286344, 60 fc rgb "#FF0000"
set object 61 rect from 2.517487, 50 to 21.699896, 60 fc rgb "#00FF00"
set object 62 rect from 2.519459, 50 to 21.706340, 60 fc rgb "#0000FF"
set object 63 rect from 2.519976, 50 to 21.744672, 60 fc rgb "#FFFF00"
set object 64 rect from 2.524502, 50 to 138.201742, 60 fc rgb "#FF00FF"
set object 65 rect from 16.043900, 50 to 143.378221, 60 fc rgb "#808080"
set object 66 rect from 16.644927, 50 to 143.900215, 60 fc rgb "#800080"
set object 67 rect from 16.705813, 50 to 143.911422, 60 fc rgb "#008080"
set object 68 rect from 16.706717, 50 to 144.087863, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
