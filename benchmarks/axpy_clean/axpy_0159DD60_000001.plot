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

set object 15 rect from 0.248739, 0 to 157.338692, 10 fc rgb "#FF0000"
set object 16 rect from 0.158376, 0 to 97.002791, 10 fc rgb "#00FF00"
set object 17 rect from 0.160275, 0 to 97.394604, 10 fc rgb "#0000FF"
set object 18 rect from 0.160688, 0 to 97.791270, 10 fc rgb "#FFFF00"
set object 19 rect from 0.161366, 0 to 98.426907, 10 fc rgb "#FF00FF"
set object 20 rect from 0.162402, 0 to 103.590258, 10 fc rgb "#808080"
set object 21 rect from 0.170946, 0 to 103.931124, 10 fc rgb "#800080"
set object 22 rect from 0.171705, 0 to 104.236208, 10 fc rgb "#008080"
set object 23 rect from 0.171970, 0 to 150.389744, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.244483, 10 to 154.908348, 20 fc rgb "#FF0000"
set object 25 rect from 0.108434, 10 to 66.784491, 20 fc rgb "#00FF00"
set object 26 rect from 0.110446, 10 to 67.215121, 20 fc rgb "#0000FF"
set object 27 rect from 0.110932, 10 to 67.627558, 20 fc rgb "#FFFF00"
set object 28 rect from 0.111648, 10 to 68.311115, 20 fc rgb "#FF00FF"
set object 29 rect from 0.112751, 10 to 73.505394, 20 fc rgb "#808080"
set object 30 rect from 0.121318, 10 to 73.853544, 20 fc rgb "#800080"
set object 31 rect from 0.122134, 10 to 74.218062, 20 fc rgb "#008080"
set object 32 rect from 0.122480, 10 to 147.839304, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.242476, 20 to 157.774185, 30 fc rgb "#FF0000"
set object 34 rect from 0.077855, 20 to 48.336422, 30 fc rgb "#00FF00"
set object 35 rect from 0.080047, 20 to 48.824669, 30 fc rgb "#0000FF"
set object 36 rect from 0.080613, 20 to 50.696406, 30 fc rgb "#FFFF00"
set object 37 rect from 0.083702, 20 to 54.052316, 30 fc rgb "#FF00FF"
set object 38 rect from 0.089252, 20 to 59.101030, 30 fc rgb "#808080"
set object 39 rect from 0.097567, 20 to 59.485567, 30 fc rgb "#800080"
set object 40 rect from 0.098434, 20 to 59.904674, 30 fc rgb "#008080"
set object 41 rect from 0.098913, 20 to 146.608057, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.240381, 30 to 158.377693, 40 fc rgb "#FF0000"
set object 43 rect from 0.041579, 30 to 27.065581, 40 fc rgb "#00FF00"
set object 44 rect from 0.045160, 30 to 28.008726, 40 fc rgb "#0000FF"
set object 45 rect from 0.046307, 30 to 30.605260, 40 fc rgb "#FFFF00"
set object 46 rect from 0.050615, 30 to 34.434260, 40 fc rgb "#FF00FF"
set object 47 rect from 0.056896, 30 to 39.754700, 40 fc rgb "#808080"
set object 48 rect from 0.065705, 30 to 40.244158, 40 fc rgb "#800080"
set object 49 rect from 0.066904, 30 to 40.945908, 40 fc rgb "#008080"
set object 50 rect from 0.067649, 30 to 144.872793, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.246458, 40 to 160.215465, 50 fc rgb "#FF0000"
set object 52 rect from 0.130537, 40 to 80.015234, 50 fc rgb "#00FF00"
set object 53 rect from 0.132260, 40 to 80.388853, 50 fc rgb "#0000FF"
set object 54 rect from 0.132650, 40 to 82.304261, 50 fc rgb "#FFFF00"
set object 55 rect from 0.135821, 40 to 85.725064, 50 fc rgb "#FF00FF"
set object 56 rect from 0.141465, 40 to 90.770741, 50 fc rgb "#808080"
set object 57 rect from 0.149797, 40 to 91.185004, 50 fc rgb "#800080"
set object 58 rect from 0.150701, 40 to 91.586523, 50 fc rgb "#008080"
set object 59 rect from 0.151118, 40 to 149.022034, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.250154, 50 to 162.396526, 60 fc rgb "#FF0000"
set object 61 rect from 0.210749, 50 to 128.827774, 60 fc rgb "#00FF00"
set object 62 rect from 0.212841, 50 to 129.333618, 60 fc rgb "#0000FF"
set object 63 rect from 0.213348, 50 to 131.311497, 60 fc rgb "#FFFF00"
set object 64 rect from 0.216640, 50 to 134.528502, 60 fc rgb "#FF00FF"
set object 65 rect from 0.221936, 50 to 139.594208, 60 fc rgb "#808080"
set object 66 rect from 0.230289, 50 to 140.013920, 60 fc rgb "#800080"
set object 67 rect from 0.231220, 50 to 140.511269, 60 fc rgb "#008080"
set object 68 rect from 0.231788, 50 to 151.431149, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
