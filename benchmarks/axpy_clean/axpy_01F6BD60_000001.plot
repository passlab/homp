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

set object 15 rect from 0.195137, 0 to 157.275636, 10 fc rgb "#FF0000"
set object 16 rect from 0.118200, 0 to 91.726947, 10 fc rgb "#00FF00"
set object 17 rect from 0.120090, 0 to 92.237053, 10 fc rgb "#0000FF"
set object 18 rect from 0.120539, 0 to 92.725703, 10 fc rgb "#FFFF00"
set object 19 rect from 0.121189, 0 to 93.444908, 10 fc rgb "#FF00FF"
set object 20 rect from 0.122134, 0 to 99.683291, 10 fc rgb "#808080"
set object 21 rect from 0.130267, 0 to 100.083867, 10 fc rgb "#800080"
set object 22 rect from 0.131046, 0 to 100.489806, 10 fc rgb "#008080"
set object 23 rect from 0.131311, 0 to 148.929444, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.191833, 10 to 155.181615, 20 fc rgb "#FF0000"
set object 25 rect from 0.066958, 10 to 52.640690, 20 fc rgb "#00FF00"
set object 26 rect from 0.069062, 10 to 53.153090, 20 fc rgb "#0000FF"
set object 27 rect from 0.069507, 10 to 53.706840, 20 fc rgb "#FFFF00"
set object 28 rect from 0.070261, 10 to 54.569267, 20 fc rgb "#FF00FF"
set object 29 rect from 0.071383, 10 to 61.038971, 20 fc rgb "#808080"
set object 30 rect from 0.079839, 10 to 61.472473, 20 fc rgb "#800080"
set object 31 rect from 0.080690, 10 to 61.969556, 20 fc rgb "#008080"
set object 32 rect from 0.081030, 10 to 146.396558, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.190103, 20 to 160.899952, 30 fc rgb "#FF0000"
set object 34 rect from 0.033269, 20 to 27.628149, 30 fc rgb "#00FF00"
set object 35 rect from 0.036515, 20 to 28.665961, 30 fc rgb "#0000FF"
set object 36 rect from 0.037559, 20 to 31.839918, 30 fc rgb "#FFFF00"
set object 37 rect from 0.041742, 20 to 36.591672, 30 fc rgb "#FF00FF"
set object 38 rect from 0.047890, 20 to 43.016180, 30 fc rgb "#808080"
set object 39 rect from 0.056388, 20 to 43.664145, 30 fc rgb "#800080"
set object 40 rect from 0.057548, 20 to 44.548017, 30 fc rgb "#008080"
set object 41 rect from 0.058289, 20 to 144.703112, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.196506, 30 to 162.957211, 40 fc rgb "#FF0000"
set object 43 rect from 0.138428, 30 to 107.044525, 40 fc rgb "#00FF00"
set object 44 rect from 0.140089, 30 to 107.559984, 40 fc rgb "#0000FF"
set object 45 rect from 0.140546, 30 to 109.811014, 40 fc rgb "#FFFF00"
set object 46 rect from 0.143488, 30 to 113.443753, 40 fc rgb "#FF00FF"
set object 47 rect from 0.148257, 30 to 119.552702, 40 fc rgb "#808080"
set object 48 rect from 0.156231, 30 to 120.055914, 40 fc rgb "#800080"
set object 49 rect from 0.157108, 30 to 120.513931, 40 fc rgb "#008080"
set object 50 rect from 0.157464, 30 to 150.141122, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.197963, 40 to 163.825733, 50 fc rgb "#FF0000"
set object 52 rect from 0.165452, 40 to 127.659173, 50 fc rgb "#00FF00"
set object 53 rect from 0.167004, 40 to 128.098804, 50 fc rgb "#0000FF"
set object 54 rect from 0.167368, 40 to 130.070271, 50 fc rgb "#FFFF00"
set object 55 rect from 0.169931, 40 to 133.721397, 50 fc rgb "#FF00FF"
set object 56 rect from 0.174699, 40 to 139.934512, 50 fc rgb "#808080"
set object 57 rect from 0.182813, 40 to 140.374154, 50 fc rgb "#800080"
set object 58 rect from 0.183631, 40 to 140.821455, 50 fc rgb "#008080"
set object 59 rect from 0.183969, 40 to 151.278512, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.193537, 50 to 161.097557, 60 fc rgb "#FF0000"
set object 61 rect from 0.089441, 50 to 69.640210, 60 fc rgb "#00FF00"
set object 62 rect from 0.091249, 50 to 70.144952, 60 fc rgb "#0000FF"
set object 63 rect from 0.091691, 50 to 72.429684, 60 fc rgb "#FFFF00"
set object 64 rect from 0.094719, 50 to 76.416274, 60 fc rgb "#FF00FF"
set object 65 rect from 0.099895, 50 to 82.586499, 60 fc rgb "#808080"
set object 66 rect from 0.107963, 50 to 83.071325, 60 fc rgb "#800080"
set object 67 rect from 0.108812, 50 to 83.583725, 60 fc rgb "#008080"
set object 68 rect from 0.109238, 50 to 147.742270, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
