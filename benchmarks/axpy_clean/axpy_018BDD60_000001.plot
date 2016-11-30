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

set object 15 rect from 0.219369, 0 to 156.076319, 10 fc rgb "#FF0000"
set object 16 rect from 0.137737, 0 to 94.713242, 10 fc rgb "#00FF00"
set object 17 rect from 0.139867, 0 to 95.252883, 10 fc rgb "#0000FF"
set object 18 rect from 0.140435, 0 to 95.692732, 10 fc rgb "#FFFF00"
set object 19 rect from 0.141142, 0 to 96.431258, 10 fc rgb "#FF00FF"
set object 20 rect from 0.142200, 0 to 102.010882, 10 fc rgb "#808080"
set object 21 rect from 0.150422, 0 to 102.409331, 10 fc rgb "#800080"
set object 22 rect from 0.151260, 0 to 102.800993, 10 fc rgb "#008080"
set object 23 rect from 0.151559, 0 to 148.519358, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.222442, 10 to 158.242316, 20 fc rgb "#FF0000"
set object 25 rect from 0.192344, 10 to 132.073703, 20 fc rgb "#00FF00"
set object 26 rect from 0.194992, 10 to 132.593650, 20 fc rgb "#0000FF"
set object 27 rect from 0.195461, 10 to 133.123783, 20 fc rgb "#FFFF00"
set object 28 rect from 0.196232, 10 to 134.027250, 20 fc rgb "#FF00FF"
set object 29 rect from 0.197578, 10 to 139.473167, 20 fc rgb "#808080"
set object 30 rect from 0.205616, 10 to 139.878403, 20 fc rgb "#800080"
set object 31 rect from 0.206445, 10 to 140.238153, 20 fc rgb "#008080"
set object 32 rect from 0.206716, 10 to 150.654151, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.216386, 20 to 158.532841, 30 fc rgb "#FF0000"
set object 34 rect from 0.076952, 20 to 53.380519, 30 fc rgb "#00FF00"
set object 35 rect from 0.078994, 20 to 54.028084, 30 fc rgb "#0000FF"
set object 36 rect from 0.079701, 20 to 56.037292, 30 fc rgb "#FFFF00"
set object 37 rect from 0.082710, 20 to 59.561554, 30 fc rgb "#FF00FF"
set object 38 rect from 0.087878, 20 to 65.087570, 30 fc rgb "#808080"
set object 39 rect from 0.096004, 20 to 65.544391, 30 fc rgb "#800080"
set object 40 rect from 0.096967, 20 to 66.024294, 30 fc rgb "#008080"
set object 41 rect from 0.097393, 20 to 146.343854, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.214358, 30 to 160.645897, 40 fc rgb "#FF0000"
set object 43 rect from 0.038042, 30 to 28.326477, 40 fc rgb "#00FF00"
set object 44 rect from 0.042459, 30 to 30.640462, 40 fc rgb "#0000FF"
set object 45 rect from 0.045310, 30 to 33.849772, 40 fc rgb "#FFFF00"
set object 46 rect from 0.050082, 30 to 38.158036, 40 fc rgb "#FF00FF"
set object 47 rect from 0.056335, 30 to 43.796042, 40 fc rgb "#808080"
set object 48 rect from 0.064648, 30 to 44.307169, 40 fc rgb "#800080"
set object 49 rect from 0.065826, 30 to 45.287336, 40 fc rgb "#008080"
set object 50 rect from 0.066840, 30 to 144.703216, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.217883, 40 to 160.132049, 50 fc rgb "#FF0000"
set object 52 rect from 0.106052, 40 to 73.060599, 50 fc rgb "#00FF00"
set object 53 rect from 0.107956, 40 to 73.471267, 50 fc rgb "#0000FF"
set object 54 rect from 0.108345, 40 to 75.553787, 50 fc rgb "#FFFF00"
set object 55 rect from 0.111444, 40 to 79.054977, 50 fc rgb "#FF00FF"
set object 56 rect from 0.116576, 40 to 85.238045, 50 fc rgb "#808080"
set object 57 rect from 0.125725, 40 to 85.785150, 50 fc rgb "#800080"
set object 58 rect from 0.126806, 40 to 86.262343, 50 fc rgb "#008080"
set object 59 rect from 0.127210, 40 to 147.550732, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.220964, 50 to 162.788842, 60 fc rgb "#FF0000"
set object 61 rect from 0.160169, 50 to 109.635024, 60 fc rgb "#00FF00"
set object 62 rect from 0.161843, 50 to 110.042981, 60 fc rgb "#0000FF"
set object 63 rect from 0.162225, 50 to 112.082736, 60 fc rgb "#FFFF00"
set object 64 rect from 0.165236, 50 to 115.661304, 60 fc rgb "#FF00FF"
set object 65 rect from 0.170514, 50 to 122.359576, 60 fc rgb "#808080"
set object 66 rect from 0.180436, 50 to 122.935184, 60 fc rgb "#800080"
set object 67 rect from 0.181510, 50 to 123.436136, 60 fc rgb "#008080"
set object 68 rect from 0.181992, 50 to 149.612891, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
