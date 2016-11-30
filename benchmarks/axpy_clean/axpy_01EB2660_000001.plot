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

set object 15 rect from 1.507671, 0 to 152.645379, 10 fc rgb "#FF0000"
set object 16 rect from 1.225370, 0 to 122.149111, 10 fc rgb "#00FF00"
set object 17 rect from 1.243371, 0 to 122.651963, 10 fc rgb "#0000FF"
set object 18 rect from 1.246450, 0 to 123.223148, 10 fc rgb "#FFFF00"
set object 19 rect from 1.252746, 0 to 124.200889, 10 fc rgb "#FF00FF"
set object 20 rect from 1.262325, 0 to 126.539493, 10 fc rgb "#808080"
set object 21 rect from 1.285993, 0 to 126.926946, 10 fc rgb "#800080"
set object 22 rect from 1.291819, 0 to 127.343347, 10 fc rgb "#008080"
set object 23 rect from 1.294073, 0 to 148.027742, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.538907, 10 to 154.942624, 20 fc rgb "#FF0000"
set object 25 rect from 1.363630, 10 to 135.808931, 20 fc rgb "#00FF00"
set object 26 rect from 1.381750, 10 to 136.246995, 20 fc rgb "#0000FF"
set object 27 rect from 1.384651, 10 to 136.716172, 20 fc rgb "#FFFF00"
set object 28 rect from 1.389284, 10 to 137.382967, 20 fc rgb "#FF00FF"
set object 29 rect from 1.396028, 10 to 139.362768, 20 fc rgb "#808080"
set object 30 rect from 1.416309, 10 to 139.728462, 20 fc rgb "#800080"
set object 31 rect from 1.421689, 10 to 140.071312, 20 fc rgb "#008080"
set object 32 rect from 1.423350, 10 to 151.169215, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.528637, 20 to 156.336671, 30 fc rgb "#FF0000"
set object 34 rect from 1.079571, 20 to 107.631968, 30 fc rgb "#00FF00"
set object 35 rect from 1.095889, 20 to 108.064419, 30 fc rgb "#0000FF"
set object 36 rect from 1.098265, 20 to 110.005822, 30 fc rgb "#FFFF00"
set object 37 rect from 1.118076, 20 to 110.965937, 30 fc rgb "#FF00FF"
set object 38 rect from 1.127746, 20 to 113.509640, 30 fc rgb "#808080"
set object 39 rect from 1.153833, 20 to 113.986400, 30 fc rgb "#800080"
set object 40 rect from 1.160720, 20 to 114.536416, 30 fc rgb "#008080"
set object 41 rect from 1.164194, 20 to 150.165086, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.496217, 30 to 153.398521, 40 fc rgb "#FF0000"
set object 43 rect from 0.615877, 30 to 61.871024, 40 fc rgb "#00FF00"
set object 44 rect from 0.631356, 30 to 62.419761, 40 fc rgb "#0000FF"
set object 45 rect from 0.634770, 30 to 64.653303, 40 fc rgb "#FFFF00"
set object 46 rect from 0.657827, 30 to 65.899949, 40 fc rgb "#FF00FF"
set object 47 rect from 0.670450, 30 to 68.134574, 40 fc rgb "#808080"
set object 48 rect from 0.692809, 30 to 68.564761, 40 fc rgb "#800080"
set object 49 rect from 0.699287, 30 to 69.078346, 40 fc rgb "#008080"
set object 50 rect from 0.702419, 30 to 146.861049, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.518158, 40 to 166.447085, 50 fc rgb "#FF0000"
set object 52 rect from 0.771676, 40 to 76.960436, 50 fc rgb "#00FF00"
set object 53 rect from 0.784069, 40 to 77.349858, 50 fc rgb "#0000FF"
set object 54 rect from 0.786321, 40 to 79.039884, 50 fc rgb "#FFFF00"
set object 55 rect from 0.803657, 40 to 79.996061, 50 fc rgb "#FF00FF"
set object 56 rect from 0.813233, 40 to 93.767147, 50 fc rgb "#808080"
set object 57 rect from 0.953612, 40 to 94.464069, 50 fc rgb "#800080"
set object 58 rect from 0.962207, 40 to 95.181966, 50 fc rgb "#008080"
set object 59 rect from 0.967614, 40 to 149.108574, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.480382, 50 to 153.523177, 60 fc rgb "#FF0000"
set object 61 rect from 0.406891, 50 to 41.876859, 60 fc rgb "#00FF00"
set object 62 rect from 0.428664, 50 to 42.584219, 60 fc rgb "#0000FF"
set object 63 rect from 0.434934, 50 to 45.911887, 60 fc rgb "#FFFF00"
set object 64 rect from 0.467689, 50 to 47.763983, 60 fc rgb "#FF00FF"
set object 65 rect from 0.485924, 50 to 49.921413, 60 fc rgb "#808080"
set object 66 rect from 0.507886, 50 to 50.405362, 60 fc rgb "#800080"
set object 67 rect from 0.515917, 50 to 51.235308, 60 fc rgb "#008080"
set object 68 rect from 0.521218, 50 to 145.109484, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
