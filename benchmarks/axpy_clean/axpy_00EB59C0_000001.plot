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

set object 15 rect from 0.165690, 0 to 158.434545, 10 fc rgb "#FF0000"
set object 16 rect from 0.141421, 0 to 134.093850, 10 fc rgb "#00FF00"
set object 17 rect from 0.143767, 0 to 134.785925, 10 fc rgb "#0000FF"
set object 18 rect from 0.144233, 0 to 135.504200, 10 fc rgb "#FFFF00"
set object 19 rect from 0.145015, 0 to 136.492755, 10 fc rgb "#FF00FF"
set object 20 rect from 0.146070, 0 to 137.796493, 10 fc rgb "#808080"
set object 21 rect from 0.147475, 0 to 138.295913, 10 fc rgb "#800080"
set object 22 rect from 0.148254, 0 to 138.794414, 10 fc rgb "#008080"
set object 23 rect from 0.148527, 0 to 154.419590, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.163990, 10 to 157.206621, 20 fc rgb "#FF0000"
set object 25 rect from 0.123772, 10 to 117.599831, 20 fc rgb "#00FF00"
set object 26 rect from 0.126221, 10 to 118.450919, 20 fc rgb "#0000FF"
set object 27 rect from 0.126770, 10 to 119.223434, 20 fc rgb "#FFFF00"
set object 28 rect from 0.127631, 10 to 120.389677, 20 fc rgb "#FF00FF"
set object 29 rect from 0.128887, 10 to 121.725217, 20 fc rgb "#808080"
set object 30 rect from 0.130309, 10 to 122.320017, 20 fc rgb "#800080"
set object 31 rect from 0.131178, 10 to 122.890540, 20 fc rgb "#008080"
set object 32 rect from 0.131524, 10 to 152.680009, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.161939, 20 to 155.355745, 30 fc rgb "#FF0000"
set object 34 rect from 0.107892, 20 to 102.650842, 30 fc rgb "#00FF00"
set object 35 rect from 0.110150, 20 to 103.368170, 30 fc rgb "#0000FF"
set object 36 rect from 0.110642, 20 to 104.206158, 30 fc rgb "#FFFF00"
set object 37 rect from 0.111542, 20 to 105.488378, 30 fc rgb "#FF00FF"
set object 38 rect from 0.112913, 20 to 106.659303, 30 fc rgb "#808080"
set object 39 rect from 0.114164, 20 to 107.258814, 30 fc rgb "#800080"
set object 40 rect from 0.115067, 20 to 107.942471, 30 fc rgb "#008080"
set object 41 rect from 0.115536, 20 to 150.966657, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.160257, 30 to 154.087600, 40 fc rgb "#FF0000"
set object 43 rect from 0.086700, 30 to 83.247265, 40 fc rgb "#00FF00"
set object 44 rect from 0.089401, 30 to 84.037563, 40 fc rgb "#0000FF"
set object 45 rect from 0.089973, 30 to 85.089723, 40 fc rgb "#FFFF00"
set object 46 rect from 0.091098, 30 to 86.346691, 40 fc rgb "#FF00FF"
set object 47 rect from 0.092443, 30 to 87.542869, 40 fc rgb "#808080"
set object 48 rect from 0.093724, 30 to 88.152636, 40 fc rgb "#800080"
set object 49 rect from 0.094664, 30 to 88.943854, 40 fc rgb "#008080"
set object 50 rect from 0.095225, 30 to 149.294416, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.158575, 40 to 152.291856, 50 fc rgb "#FF0000"
set object 52 rect from 0.057143, 40 to 55.392818, 50 fc rgb "#00FF00"
set object 53 rect from 0.059589, 40 to 56.195241, 50 fc rgb "#0000FF"
set object 54 rect from 0.060206, 40 to 57.016394, 50 fc rgb "#FFFF00"
set object 55 rect from 0.061093, 40 to 58.329497, 50 fc rgb "#FF00FF"
set object 56 rect from 0.062515, 40 to 59.568682, 50 fc rgb "#808080"
set object 57 rect from 0.063837, 40 to 60.101772, 50 fc rgb "#800080"
set object 58 rect from 0.064652, 40 to 60.742422, 50 fc rgb "#008080"
set object 59 rect from 0.065084, 40 to 147.723213, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.156423, 50 to 153.401992, 60 fc rgb "#FF0000"
set object 61 rect from 0.031554, 50 to 32.436245, 60 fc rgb "#00FF00"
set object 62 rect from 0.035143, 50 to 33.764288, 60 fc rgb "#0000FF"
set object 63 rect from 0.036252, 50 to 36.277304, 60 fc rgb "#FFFF00"
set object 64 rect from 0.038971, 50 to 38.152457, 60 fc rgb "#FF00FF"
set object 65 rect from 0.040935, 50 to 39.697487, 60 fc rgb "#808080"
set object 66 rect from 0.042592, 50 to 40.429782, 60 fc rgb "#800080"
set object 67 rect from 0.043832, 50 to 41.682095, 60 fc rgb "#008080"
set object 68 rect from 0.044702, 50 to 145.322440, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
