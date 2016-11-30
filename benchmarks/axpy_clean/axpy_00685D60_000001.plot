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

set object 15 rect from 0.212837, 0 to 156.176964, 10 fc rgb "#FF0000"
set object 16 rect from 0.103950, 0 to 73.487273, 10 fc rgb "#00FF00"
set object 17 rect from 0.106096, 0 to 73.980828, 10 fc rgb "#0000FF"
set object 18 rect from 0.106562, 0 to 74.512626, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107349, 0 to 75.341239, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108540, 0 to 81.870805, 10 fc rgb "#808080"
set object 21 rect from 0.117920, 0 to 82.310143, 10 fc rgb "#800080"
set object 22 rect from 0.118824, 0 to 82.768250, 10 fc rgb "#008080"
set object 23 rect from 0.119189, 0 to 147.459068, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.216278, 10 to 158.226987, 20 fc rgb "#FF0000"
set object 25 rect from 0.157827, 10 to 110.711961, 20 fc rgb "#00FF00"
set object 26 rect from 0.159638, 10 to 111.171457, 20 fc rgb "#0000FF"
set object 27 rect from 0.160043, 10 to 111.671963, 20 fc rgb "#FFFF00"
set object 28 rect from 0.160776, 10 to 112.442889, 20 fc rgb "#FF00FF"
set object 29 rect from 0.161884, 10 to 118.703421, 20 fc rgb "#808080"
set object 30 rect from 0.170893, 10 to 119.128858, 20 fc rgb "#800080"
set object 31 rect from 0.171751, 10 to 119.520920, 20 fc rgb "#008080"
set object 32 rect from 0.172067, 10 to 149.934507, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.214682, 20 to 162.081618, 30 fc rgb "#FF0000"
set object 34 rect from 0.128267, 20 to 90.233485, 30 fc rgb "#00FF00"
set object 35 rect from 0.130186, 20 to 90.712444, 30 fc rgb "#0000FF"
set object 36 rect from 0.130612, 20 to 92.859765, 30 fc rgb "#FFFF00"
set object 37 rect from 0.133716, 20 to 96.843681, 30 fc rgb "#FF00FF"
set object 38 rect from 0.139447, 20 to 103.107000, 30 fc rgb "#808080"
set object 39 rect from 0.148486, 20 to 103.642963, 30 fc rgb "#800080"
set object 40 rect from 0.149537, 20 to 104.157370, 30 fc rgb "#008080"
set object 41 rect from 0.149979, 20 to 148.732592, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.217765, 30 to 164.015511, 40 fc rgb "#FF0000"
set object 43 rect from 0.181115, 30 to 126.867983, 40 fc rgb "#00FF00"
set object 44 rect from 0.182867, 30 to 127.360150, 40 fc rgb "#0000FF"
set object 45 rect from 0.183332, 30 to 129.599932, 40 fc rgb "#FFFF00"
set object 46 rect from 0.186553, 30 to 133.295351, 40 fc rgb "#FF00FF"
set object 47 rect from 0.191897, 30 to 139.535032, 40 fc rgb "#808080"
set object 48 rect from 0.200852, 30 to 140.027209, 40 fc rgb "#800080"
set object 49 rect from 0.201839, 30 to 140.492257, 40 fc rgb "#008080"
set object 50 rect from 0.202249, 30 to 151.041186, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.209227, 40 to 160.570334, 50 fc rgb "#FF0000"
set object 52 rect from 0.034960, 40 to 26.508220, 50 fc rgb "#00FF00"
set object 53 rect from 0.038617, 40 to 27.503679, 50 fc rgb "#0000FF"
set object 54 rect from 0.039705, 40 to 30.654103, 50 fc rgb "#FFFF00"
set object 55 rect from 0.044261, 40 to 35.260880, 50 fc rgb "#FF00FF"
set object 56 rect from 0.050858, 40 to 41.711886, 50 fc rgb "#808080"
set object 57 rect from 0.060149, 40 to 42.282602, 50 fc rgb "#800080"
set object 58 rect from 0.061411, 40 to 43.159196, 50 fc rgb "#008080"
set object 59 rect from 0.062229, 40 to 144.859896, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.211101, 50 to 159.629092, 60 fc rgb "#FF0000"
set object 61 rect from 0.072378, 50 to 51.595556, 60 fc rgb "#00FF00"
set object 62 rect from 0.074589, 50 to 52.063390, 60 fc rgb "#0000FF"
set object 63 rect from 0.075013, 50 to 54.399103, 60 fc rgb "#FFFF00"
set object 64 rect from 0.078378, 50 to 58.251631, 60 fc rgb "#FF00FF"
set object 65 rect from 0.083933, 50 to 64.490618, 60 fc rgb "#808080"
set object 66 rect from 0.092901, 50 to 65.002248, 60 fc rgb "#800080"
set object 67 rect from 0.093921, 50 to 65.497201, 60 fc rgb "#008080"
set object 68 rect from 0.094381, 50 to 146.259245, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
