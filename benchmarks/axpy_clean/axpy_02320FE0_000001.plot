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

set object 15 rect from 0.334630, 0 to 158.103515, 10 fc rgb "#FF0000"
set object 16 rect from 0.150591, 0 to 67.233706, 10 fc rgb "#00FF00"
set object 17 rect from 0.153196, 0 to 67.580002, 10 fc rgb "#0000FF"
set object 18 rect from 0.153697, 0 to 67.951378, 10 fc rgb "#FFFF00"
set object 19 rect from 0.154576, 0 to 68.493922, 10 fc rgb "#FF00FF"
set object 20 rect from 0.155811, 0 to 78.191079, 10 fc rgb "#808080"
set object 21 rect from 0.177855, 0 to 78.460811, 10 fc rgb "#800080"
set object 22 rect from 0.178706, 0 to 78.739344, 10 fc rgb "#008080"
set object 23 rect from 0.179079, 0 to 146.842528, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.339937, 10 to 160.141685, 20 fc rgb "#FF0000"
set object 25 rect from 0.294552, 10 to 130.424069, 20 fc rgb "#00FF00"
set object 26 rect from 0.296756, 10 to 130.706122, 20 fc rgb "#0000FF"
set object 27 rect from 0.297164, 10 to 131.032177, 20 fc rgb "#FFFF00"
set object 28 rect from 0.297912, 10 to 131.505198, 20 fc rgb "#FF00FF"
set object 29 rect from 0.298995, 10 to 141.069908, 20 fc rgb "#808080"
set object 30 rect from 0.320765, 10 to 141.330840, 20 fc rgb "#800080"
set object 31 rect from 0.321574, 10 to 141.588692, 20 fc rgb "#008080"
set object 32 rect from 0.321900, 10 to 149.293438, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.329801, 20 to 163.402672, 30 fc rgb "#FF0000"
set object 34 rect from 0.039664, 20 to 19.032617, 30 fc rgb "#00FF00"
set object 35 rect from 0.043923, 20 to 19.745009, 30 fc rgb "#0000FF"
set object 36 rect from 0.045014, 20 to 21.772180, 30 fc rgb "#FFFF00"
set object 37 rect from 0.049645, 20 to 27.714648, 30 fc rgb "#FF00FF"
set object 38 rect from 0.063115, 20 to 37.352841, 30 fc rgb "#808080"
set object 39 rect from 0.085032, 20 to 37.809141, 30 fc rgb "#800080"
set object 40 rect from 0.086562, 20 to 38.455091, 30 fc rgb "#008080"
set object 41 rect from 0.087579, 20 to 144.587427, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.338231, 30 to 165.747980, 40 fc rgb "#FF0000"
set object 43 rect from 0.243302, 30 to 107.791211, 40 fc rgb "#00FF00"
set object 44 rect from 0.245341, 30 to 108.120345, 40 fc rgb "#0000FF"
set object 45 rect from 0.245844, 30 to 109.450085, 40 fc rgb "#FFFF00"
set object 46 rect from 0.248865, 30 to 115.130741, 40 fc rgb "#FF00FF"
set object 47 rect from 0.261763, 30 to 124.658050, 40 fc rgb "#808080"
set object 48 rect from 0.283435, 30 to 125.043508, 40 fc rgb "#800080"
set object 49 rect from 0.284561, 30 to 125.338761, 40 fc rgb "#008080"
set object 50 rect from 0.285092, 30 to 148.556846, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.336526, 40 to 164.644409, 50 fc rgb "#FF0000"
set object 52 rect from 0.188593, 40 to 83.786811, 50 fc rgb "#00FF00"
set object 53 rect from 0.190891, 40 to 84.131786, 50 fc rgb "#0000FF"
set object 54 rect from 0.191311, 40 to 85.513009, 50 fc rgb "#FFFF00"
set object 55 rect from 0.194465, 40 to 90.868050, 50 fc rgb "#FF00FF"
set object 56 rect from 0.206636, 40 to 100.341235, 50 fc rgb "#808080"
set object 57 rect from 0.228191, 40 to 100.740773, 50 fc rgb "#800080"
set object 58 rect from 0.229325, 40 to 101.040867, 50 fc rgb "#008080"
set object 59 rect from 0.229824, 40 to 147.752929, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.332326, 50 to 163.264066, 60 fc rgb "#FF0000"
set object 61 rect from 0.099380, 50 to 44.687532, 60 fc rgb "#00FF00"
set object 62 rect from 0.101929, 50 to 45.127991, 60 fc rgb "#0000FF"
set object 63 rect from 0.102672, 50 to 46.619218, 60 fc rgb "#FFFF00"
set object 64 rect from 0.106115, 50 to 52.115505, 60 fc rgb "#FF00FF"
set object 65 rect from 0.118553, 50 to 61.667015, 60 fc rgb "#808080"
set object 66 rect from 0.140280, 50 to 62.068753, 60 fc rgb "#800080"
set object 67 rect from 0.141463, 50 to 62.371487, 60 fc rgb "#008080"
set object 68 rect from 0.141900, 50 to 145.798802, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
