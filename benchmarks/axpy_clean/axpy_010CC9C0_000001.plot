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

set object 15 rect from 0.133502, 0 to 159.940599, 10 fc rgb "#FF0000"
set object 16 rect from 0.112053, 0 to 132.899012, 10 fc rgb "#00FF00"
set object 17 rect from 0.114224, 0 to 133.663592, 10 fc rgb "#0000FF"
set object 18 rect from 0.114635, 0 to 134.567079, 10 fc rgb "#FFFF00"
set object 19 rect from 0.115394, 0 to 135.744881, 10 fc rgb "#FF00FF"
set object 20 rect from 0.116405, 0 to 137.225013, 10 fc rgb "#808080"
set object 21 rect from 0.117672, 0 to 137.829672, 10 fc rgb "#800080"
set object 22 rect from 0.118444, 0 to 138.456510, 10 fc rgb "#008080"
set object 23 rect from 0.118730, 0 to 155.204877, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.131938, 10 to 158.663575, 20 fc rgb "#FF0000"
set object 25 rect from 0.095229, 10 to 113.357306, 20 fc rgb "#00FF00"
set object 26 rect from 0.097544, 10 to 114.285307, 20 fc rgb "#0000FF"
set object 27 rect from 0.098018, 10 to 115.248326, 20 fc rgb "#FFFF00"
set object 28 rect from 0.098884, 10 to 116.734295, 20 fc rgb "#FF00FF"
set object 29 rect from 0.100147, 10 to 118.291467, 20 fc rgb "#808080"
set object 30 rect from 0.101472, 10 to 119.016359, 20 fc rgb "#800080"
set object 31 rect from 0.102328, 10 to 119.717904, 20 fc rgb "#008080"
set object 32 rect from 0.102695, 10 to 153.223975, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.130187, 20 to 156.256610, 30 fc rgb "#FF0000"
set object 34 rect from 0.080205, 20 to 95.669267, 30 fc rgb "#00FF00"
set object 35 rect from 0.082304, 20 to 96.406997, 30 fc rgb "#0000FF"
set object 36 rect from 0.082722, 20 to 97.393362, 30 fc rgb "#FFFF00"
set object 37 rect from 0.083547, 20 to 98.694898, 30 fc rgb "#FF00FF"
set object 38 rect from 0.084660, 20 to 100.073474, 30 fc rgb "#808080"
set object 39 rect from 0.085845, 20 to 100.776187, 30 fc rgb "#800080"
set object 40 rect from 0.086698, 20 to 101.454386, 30 fc rgb "#008080"
set object 41 rect from 0.087058, 20 to 151.421670, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.128645, 30 to 154.485824, 40 fc rgb "#FF0000"
set object 43 rect from 0.063857, 30 to 76.629497, 40 fc rgb "#00FF00"
set object 44 rect from 0.066080, 30 to 77.425594, 40 fc rgb "#0000FF"
set object 45 rect from 0.066451, 30 to 78.505342, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067366, 30 to 79.754350, 40 fc rgb "#FF00FF"
set object 47 rect from 0.068437, 30 to 81.111914, 40 fc rgb "#808080"
set object 48 rect from 0.069600, 30 to 81.861319, 40 fc rgb "#800080"
set object 49 rect from 0.070491, 30 to 82.564031, 40 fc rgb "#008080"
set object 50 rect from 0.070846, 30 to 149.513141, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.127097, 40 to 152.918144, 50 fc rgb "#FF0000"
set object 52 rect from 0.047579, 40 to 58.035635, 50 fc rgb "#00FF00"
set object 53 rect from 0.050161, 40 to 58.887762, 50 fc rgb "#0000FF"
set object 54 rect from 0.050571, 40 to 59.925488, 50 fc rgb "#FFFF00"
set object 55 rect from 0.051448, 40 to 61.402118, 50 fc rgb "#FF00FF"
set object 56 rect from 0.052751, 40 to 62.862406, 50 fc rgb "#808080"
set object 57 rect from 0.053990, 40 to 63.589631, 50 fc rgb "#800080"
set object 58 rect from 0.054892, 40 to 64.384559, 50 fc rgb "#008080"
set object 59 rect from 0.055286, 40 to 147.679318, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.125340, 50 to 154.233688, 60 fc rgb "#FF0000"
set object 61 rect from 0.025322, 50 to 33.000632, 60 fc rgb "#00FF00"
set object 62 rect from 0.028810, 50 to 34.919668, 60 fc rgb "#0000FF"
set object 63 rect from 0.030051, 50 to 37.342975, 60 fc rgb "#FFFF00"
set object 64 rect from 0.032161, 50 to 39.410257, 60 fc rgb "#FF00FF"
set object 65 rect from 0.033884, 50 to 41.119178, 60 fc rgb "#808080"
set object 66 rect from 0.035381, 50 to 42.077528, 60 fc rgb "#800080"
set object 67 rect from 0.036621, 50 to 43.389570, 60 fc rgb "#008080"
set object 68 rect from 0.037294, 50 to 145.187140, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
