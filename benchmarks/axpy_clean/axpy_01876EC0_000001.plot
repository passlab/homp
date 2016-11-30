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

set object 15 rect from 0.276024, 0 to 156.005620, 10 fc rgb "#FF0000"
set object 16 rect from 0.124046, 0 to 66.736455, 10 fc rgb "#00FF00"
set object 17 rect from 0.125693, 0 to 67.114340, 10 fc rgb "#0000FF"
set object 18 rect from 0.126211, 0 to 67.391106, 10 fc rgb "#FFFF00"
set object 19 rect from 0.126720, 0 to 67.836054, 10 fc rgb "#FF00FF"
set object 20 rect from 0.127555, 0 to 75.910600, 10 fc rgb "#808080"
set object 21 rect from 0.142766, 0 to 76.183630, 10 fc rgb "#800080"
set object 22 rect from 0.143463, 0 to 76.427395, 10 fc rgb "#008080"
set object 23 rect from 0.143695, 0 to 146.621789, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.274357, 10 to 155.267409, 20 fc rgb "#FF0000"
set object 25 rect from 0.096575, 10 to 52.301165, 20 fc rgb "#00FF00"
set object 26 rect from 0.098572, 10 to 52.668946, 20 fc rgb "#0000FF"
set object 27 rect from 0.099076, 10 to 53.059607, 20 fc rgb "#FFFF00"
set object 28 rect from 0.099832, 10 to 53.610466, 20 fc rgb "#FF00FF"
set object 29 rect from 0.100865, 10 to 61.629654, 20 fc rgb "#808080"
set object 30 rect from 0.115924, 10 to 61.936755, 20 fc rgb "#800080"
set object 31 rect from 0.116724, 10 to 62.220436, 20 fc rgb "#008080"
set object 32 rect from 0.117019, 10 to 145.680793, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.272660, 20 to 170.249854, 30 fc rgb "#FF0000"
set object 34 rect from 0.034909, 20 to 20.334050, 30 fc rgb "#00FF00"
set object 35 rect from 0.038606, 20 to 21.008924, 30 fc rgb "#0000FF"
set object 36 rect from 0.039593, 20 to 23.285290, 30 fc rgb "#FFFF00"
set object 37 rect from 0.043961, 20 to 30.306014, 30 fc rgb "#FF00FF"
set object 38 rect from 0.057058, 20 to 45.371996, 30 fc rgb "#808080"
set object 39 rect from 0.085379, 20 to 45.929777, 30 fc rgb "#800080"
set object 40 rect from 0.086813, 20 to 46.631799, 30 fc rgb "#008080"
set object 41 rect from 0.087731, 20 to 144.548201, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.279381, 30 to 163.982242, 40 fc rgb "#FF0000"
set object 43 rect from 0.196976, 30 to 105.451788, 40 fc rgb "#00FF00"
set object 44 rect from 0.198429, 30 to 105.765805, 40 fc rgb "#0000FF"
set object 45 rect from 0.198816, 30 to 107.103847, 40 fc rgb "#FFFF00"
set object 46 rect from 0.201332, 30 to 112.722122, 40 fc rgb "#FF00FF"
set object 47 rect from 0.211894, 30 to 120.600267, 40 fc rgb "#808080"
set object 48 rect from 0.226715, 30 to 121.069167, 40 fc rgb "#800080"
set object 49 rect from 0.227805, 30 to 121.377331, 40 fc rgb "#008080"
set object 50 rect from 0.228167, 30 to 148.413290, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.277832, 40 to 163.851311, 50 fc rgb "#FF0000"
set object 52 rect from 0.151705, 40 to 81.425622, 50 fc rgb "#00FF00"
set object 53 rect from 0.153303, 40 to 81.882809, 50 fc rgb "#0000FF"
set object 54 rect from 0.153945, 40 to 83.435343, 50 fc rgb "#FFFF00"
set object 55 rect from 0.156883, 40 to 89.352732, 50 fc rgb "#FF00FF"
set object 56 rect from 0.168014, 40 to 97.287297, 50 fc rgb "#808080"
set object 57 rect from 0.182908, 40 to 97.769506, 50 fc rgb "#800080"
set object 58 rect from 0.184061, 40 to 98.131425, 50 fc rgb "#008080"
set object 59 rect from 0.184477, 40 to 147.461658, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.280667, 50 to 164.659781, 60 fc rgb "#FF0000"
set object 61 rect from 0.235040, 50 to 125.729426, 60 fc rgb "#00FF00"
set object 62 rect from 0.236517, 50 to 126.043974, 60 fc rgb "#0000FF"
set object 63 rect from 0.236916, 50 to 127.391057, 60 fc rgb "#FFFF00"
set object 64 rect from 0.239448, 50 to 133.006144, 60 fc rgb "#FF00FF"
set object 65 rect from 0.250011, 50 to 140.839051, 60 fc rgb "#808080"
set object 66 rect from 0.264732, 50 to 141.331371, 60 fc rgb "#800080"
set object 67 rect from 0.265875, 50 to 141.625164, 60 fc rgb "#008080"
set object 68 rect from 0.266194, 50 to 149.155759, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
