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

set object 15 rect from 0.354854, 0 to 155.497284, 10 fc rgb "#FF0000"
set object 16 rect from 0.077077, 0 to 32.892494, 10 fc rgb "#00FF00"
set object 17 rect from 0.079727, 0 to 33.170246, 10 fc rgb "#0000FF"
set object 18 rect from 0.080142, 0 to 33.523032, 10 fc rgb "#FFFF00"
set object 19 rect from 0.080983, 0 to 33.977800, 10 fc rgb "#FF00FF"
set object 20 rect from 0.082119, 0 to 41.364349, 10 fc rgb "#808080"
set object 21 rect from 0.099906, 0 to 41.613911, 10 fc rgb "#800080"
set object 22 rect from 0.100848, 0 to 41.873008, 10 fc rgb "#008080"
set object 23 rect from 0.101158, 0 to 146.642377, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.351235, 10 to 155.641136, 20 fc rgb "#FF0000"
set object 25 rect from 0.037300, 10 to 16.753848, 20 fc rgb "#00FF00"
set object 26 rect from 0.041111, 10 to 17.355783, 20 fc rgb "#0000FF"
set object 27 rect from 0.042002, 10 to 18.097835, 20 fc rgb "#FFFF00"
set object 28 rect from 0.043809, 10 to 18.825796, 20 fc rgb "#FF00FF"
set object 29 rect from 0.045541, 10 to 26.964349, 20 fc rgb "#808080"
set object 30 rect from 0.065265, 10 to 27.282313, 20 fc rgb "#800080"
set object 31 rect from 0.066375, 10 to 27.608153, 20 fc rgb "#008080"
set object 32 rect from 0.066719, 10 to 144.914512, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.360227, 20 to 164.353438, 30 fc rgb "#FF0000"
set object 34 rect from 0.185103, 20 to 77.843991, 30 fc rgb "#00FF00"
set object 35 rect from 0.188192, 20 to 78.226626, 30 fc rgb "#0000FF"
set object 36 rect from 0.188846, 20 to 79.975220, 30 fc rgb "#FFFF00"
set object 37 rect from 0.193131, 20 to 84.774944, 30 fc rgb "#FF00FF"
set object 38 rect from 0.204622, 20 to 92.790375, 30 fc rgb "#808080"
set object 39 rect from 0.223953, 20 to 93.239336, 30 fc rgb "#800080"
set object 40 rect from 0.225323, 20 to 93.729343, 30 fc rgb "#008080"
set object 41 rect from 0.226226, 20 to 149.000368, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.362412, 30 to 164.913082, 40 fc rgb "#FF0000"
set object 43 rect from 0.255979, 30 to 106.813127, 40 fc rgb "#00FF00"
set object 44 rect from 0.258004, 30 to 107.253387, 40 fc rgb "#0000FF"
set object 45 rect from 0.258834, 30 to 108.776048, 40 fc rgb "#FFFF00"
set object 46 rect from 0.262533, 30 to 113.565409, 40 fc rgb "#FF00FF"
set object 47 rect from 0.274062, 30 to 121.418333, 40 fc rgb "#808080"
set object 48 rect from 0.293007, 30 to 121.787288, 40 fc rgb "#800080"
set object 49 rect from 0.294189, 30 to 122.069185, 40 fc rgb "#008080"
set object 50 rect from 0.294578, 30 to 149.958822, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.358048, 40 to 162.501612, 50 fc rgb "#FF0000"
set object 52 rect from 0.112459, 40 to 47.535421, 50 fc rgb "#00FF00"
set object 53 rect from 0.115040, 40 to 47.862506, 50 fc rgb "#0000FF"
set object 54 rect from 0.115566, 40 to 49.446109, 50 fc rgb "#FFFF00"
set object 55 rect from 0.119422, 40 to 53.623171, 50 fc rgb "#FF00FF"
set object 56 rect from 0.129482, 40 to 61.527913, 50 fc rgb "#808080"
set object 57 rect from 0.148595, 40 to 61.946615, 50 fc rgb "#800080"
set object 58 rect from 0.149840, 40 to 62.438277, 50 fc rgb "#008080"
set object 59 rect from 0.150745, 40 to 147.974759, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.363916, 50 to 164.487747, 60 fc rgb "#FF0000"
set object 61 rect from 0.305176, 50 to 127.183970, 60 fc rgb "#00FF00"
set object 62 rect from 0.307139, 50 to 127.501107, 60 fc rgb "#0000FF"
set object 63 rect from 0.307676, 50 to 128.884063, 60 fc rgb "#FFFF00"
set object 64 rect from 0.311017, 50 to 132.907324, 60 fc rgb "#FF00FF"
set object 65 rect from 0.320720, 50 to 140.754030, 60 fc rgb "#808080"
set object 66 rect from 0.339652, 50 to 141.100184, 60 fc rgb "#800080"
set object 67 rect from 0.340780, 50 to 141.373792, 60 fc rgb "#008080"
set object 68 rect from 0.341151, 50 to 150.661493, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
