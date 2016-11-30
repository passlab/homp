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

set object 15 rect from 0.345961, 0 to 162.157722, 10 fc rgb "#FF0000"
set object 16 rect from 0.096402, 0 to 41.811884, 10 fc rgb "#00FF00"
set object 17 rect from 0.099208, 0 to 42.173236, 10 fc rgb "#0000FF"
set object 18 rect from 0.099809, 0 to 42.616252, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100883, 0 to 43.194668, 10 fc rgb "#FF00FF"
set object 20 rect from 0.102250, 0 to 57.635628, 10 fc rgb "#808080"
set object 21 rect from 0.136408, 0 to 57.984708, 10 fc rgb "#800080"
set object 22 rect from 0.137474, 0 to 58.272436, 10 fc rgb "#008080"
set object 23 rect from 0.137848, 0 to 146.008168, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.347956, 10 to 157.464379, 20 fc rgb "#FF0000"
set object 25 rect from 0.149424, 10 to 64.159425, 20 fc rgb "#00FF00"
set object 26 rect from 0.152019, 10 to 64.466194, 20 fc rgb "#0000FF"
set object 27 rect from 0.152471, 10 to 64.836008, 20 fc rgb "#FFFF00"
set object 28 rect from 0.153344, 10 to 65.272677, 20 fc rgb "#FF00FF"
set object 29 rect from 0.154383, 10 to 74.478688, 20 fc rgb "#808080"
set object 30 rect from 0.176172, 10 to 74.722833, 20 fc rgb "#800080"
set object 31 rect from 0.176983, 10 to 74.945399, 20 fc rgb "#008080"
set object 32 rect from 0.177255, 10 to 146.886584, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.349883, 20 to 164.255593, 30 fc rgb "#FF0000"
set object 34 rect from 0.186888, 20 to 80.039868, 30 fc rgb "#00FF00"
set object 35 rect from 0.189514, 20 to 80.310670, 30 fc rgb "#0000FF"
set object 36 rect from 0.189910, 20 to 81.717997, 30 fc rgb "#FFFF00"
set object 37 rect from 0.193276, 20 to 87.010490, 30 fc rgb "#FF00FF"
set object 38 rect from 0.205757, 20 to 96.185612, 30 fc rgb "#808080"
set object 39 rect from 0.227437, 20 to 96.560927, 30 fc rgb "#800080"
set object 40 rect from 0.228608, 20 to 96.878697, 30 fc rgb "#008080"
set object 41 rect from 0.229091, 20 to 147.767961, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.351712, 30 to 165.285064, 40 fc rgb "#FF0000"
set object 43 rect from 0.237501, 30 to 101.232691, 40 fc rgb "#00FF00"
set object 44 rect from 0.239601, 30 to 101.533535, 40 fc rgb "#0000FF"
set object 45 rect from 0.240103, 30 to 102.940439, 40 fc rgb "#FFFF00"
set object 46 rect from 0.243410, 30 to 108.440688, 40 fc rgb "#FF00FF"
set object 47 rect from 0.256399, 30 to 117.634428, 40 fc rgb "#808080"
set object 48 rect from 0.278136, 30 to 118.016514, 40 fc rgb "#800080"
set object 49 rect from 0.279310, 30 to 118.359248, 40 fc rgb "#008080"
set object 50 rect from 0.279843, 30 to 148.518591, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.353213, 40 to 165.329916, 50 fc rgb "#FF0000"
set object 52 rect from 0.293018, 40 to 124.799689, 50 fc rgb "#00FF00"
set object 53 rect from 0.295291, 40 to 125.065837, 50 fc rgb "#0000FF"
set object 54 rect from 0.295702, 40 to 126.399962, 50 fc rgb "#FFFF00"
set object 55 rect from 0.298844, 40 to 131.370878, 50 fc rgb "#FF00FF"
set object 56 rect from 0.310599, 40 to 140.619625, 50 fc rgb "#808080"
set object 57 rect from 0.332451, 40 to 140.981400, 50 fc rgb "#800080"
set object 58 rect from 0.333581, 40 to 141.256434, 50 fc rgb "#008080"
set object 59 rect from 0.333950, 40 to 149.250180, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.343724, 50 to 163.559968, 60 fc rgb "#FF0000"
set object 61 rect from 0.036109, 50 to 16.877757, 60 fc rgb "#00FF00"
set object 62 rect from 0.040412, 50 to 17.509911, 60 fc rgb "#0000FF"
set object 63 rect from 0.041527, 50 to 20.010602, 60 fc rgb "#FFFF00"
set object 64 rect from 0.047471, 50 to 25.605632, 60 fc rgb "#FF00FF"
set object 65 rect from 0.060655, 50 to 34.957199, 60 fc rgb "#808080"
set object 66 rect from 0.082832, 50 to 35.450568, 60 fc rgb "#800080"
set object 67 rect from 0.084770, 50 to 36.172002, 60 fc rgb "#008080"
set object 68 rect from 0.085646, 50 to 144.816638, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
