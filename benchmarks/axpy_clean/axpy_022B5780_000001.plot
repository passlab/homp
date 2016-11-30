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

set object 15 rect from 9.094947, 0 to 145.920277, 10 fc rgb "#FF0000"
set object 16 rect from 6.429088, 0 to 102.957245, 10 fc rgb "#00FF00"
set object 17 rect from 6.456228, 0 to 103.080992, 10 fc rgb "#0000FF"
set object 18 rect from 6.461010, 0 to 103.220998, 10 fc rgb "#FFFF00"
set object 19 rect from 6.469972, 0 to 103.389280, 10 fc rgb "#FF00FF"
set object 20 rect from 6.480285, 0 to 103.789896, 10 fc rgb "#808080"
set object 21 rect from 6.505572, 0 to 103.870111, 10 fc rgb "#800080"
set object 22 rect from 6.512439, 0 to 103.946003, 10 fc rgb "#008080"
set object 23 rect from 6.515366, 0 to 145.047207, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 9.047090, 10 to 144.908095, 20 fc rgb "#FF0000"
set object 25 rect from 0.287214, 10 to 4.787407, 20 fc rgb "#00FF00"
set object 26 rect from 0.302319, 10 to 4.864751, 20 fc rgb "#0000FF"
set object 27 rect from 0.305625, 10 to 4.951716, 20 fc rgb "#FFFF00"
set object 28 rect from 0.311335, 10 to 5.080824, 20 fc rgb "#FF00FF"
set object 29 rect from 0.319352, 10 to 5.347002, 20 fc rgb "#808080"
set object 30 rect from 0.336044, 10 to 5.410782, 20 fc rgb "#800080"
set object 31 rect from 0.341697, 10 to 5.475918, 20 fc rgb "#008080"
set object 32 rect from 0.344043, 10 to 144.244684, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 9.107600, 20 to 146.446617, 30 fc rgb "#FF0000"
set object 34 rect from 8.864158, 20 to 141.787105, 30 fc rgb "#00FF00"
set object 35 rect from 8.888547, 20 to 141.884650, 30 fc rgb "#0000FF"
set object 36 rect from 8.892757, 20 to 142.313094, 30 fc rgb "#FFFF00"
set object 37 rect from 8.919772, 20 to 142.549369, 30 fc rgb "#FF00FF"
set object 38 rect from 8.934497, 20 to 142.926320, 30 fc rgb "#808080"
set object 39 rect from 8.958130, 20 to 143.009711, 30 fc rgb "#800080"
set object 40 rect from 8.965792, 20 to 143.136345, 30 fc rgb "#008080"
set object 41 rect from 8.971222, 20 to 145.265195, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 9.122227, 30 to 185.195271, 40 fc rgb "#FF0000"
set object 43 rect from 0.502496, 30 to 8.176363, 40 fc rgb "#00FF00"
set object 44 rect from 0.514706, 30 to 8.236903, 40 fc rgb "#0000FF"
set object 45 rect from 0.517058, 30 to 8.413914, 40 fc rgb "#FFFF00"
set object 46 rect from 0.528115, 30 to 8.544761, 40 fc rgb "#FF00FF"
set object 47 rect from 0.536235, 30 to 47.843515, 40 fc rgb "#808080"
set object 48 rect from 2.998809, 30 to 47.877871, 40 fc rgb "#800080"
set object 49 rect from 3.001267, 30 to 47.910008, 40 fc rgb "#008080"
set object 50 rect from 3.002693, 30 to 145.493540, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 9.080324, 40 to 145.513518, 50 fc rgb "#FF0000"
set object 52 rect from 0.392470, 40 to 6.437453, 50 fc rgb "#00FF00"
set object 53 rect from 0.405730, 40 to 6.503371, 50 fc rgb "#0000FF"
set object 54 rect from 0.408304, 40 to 6.697088, 50 fc rgb "#FFFF00"
set object 55 rect from 0.420767, 40 to 6.854057, 50 fc rgb "#FF00FF"
set object 56 rect from 0.430466, 40 to 7.063907, 50 fc rgb "#808080"
set object 57 rect from 0.443646, 40 to 7.138203, 50 fc rgb "#800080"
set object 58 rect from 0.449874, 40 to 7.213073, 50 fc rgb "#008080"
set object 59 rect from 0.452802, 40 to 144.809911, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 9.064117, 50 to 145.436079, 60 fc rgb "#FF0000"
set object 61 rect from 0.140170, 50 to 2.541178, 60 fc rgb "#00FF00"
set object 62 rect from 0.162299, 50 to 2.637255, 60 fc rgb "#0000FF"
set object 63 rect from 0.166119, 50 to 2.911699, 60 fc rgb "#FFFF00"
set object 64 rect from 0.183501, 50 to 3.120096, 60 fc rgb "#FF00FF"
set object 65 rect from 0.196350, 50 to 3.354361, 60 fc rgb "#808080"
set object 66 rect from 0.211088, 50 to 3.431353, 60 fc rgb "#800080"
set object 67 rect from 0.218671, 50 to 3.556838, 60 fc rgb "#008080"
set object 68 rect from 0.223767, 50 to 144.547531, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
