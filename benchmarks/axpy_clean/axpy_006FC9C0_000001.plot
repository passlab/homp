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

set object 15 rect from 0.120025, 0 to 161.638373, 10 fc rgb "#FF0000"
set object 16 rect from 0.099255, 0 to 132.458898, 10 fc rgb "#00FF00"
set object 17 rect from 0.101494, 0 to 133.215803, 10 fc rgb "#0000FF"
set object 18 rect from 0.101857, 0 to 134.196650, 10 fc rgb "#FFFF00"
set object 19 rect from 0.102598, 0 to 135.527185, 10 fc rgb "#FF00FF"
set object 20 rect from 0.103612, 0 to 137.160185, 10 fc rgb "#808080"
set object 21 rect from 0.104861, 0 to 137.876501, 10 fc rgb "#800080"
set object 22 rect from 0.105653, 0 to 138.578416, 10 fc rgb "#008080"
set object 23 rect from 0.105939, 0 to 156.427657, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.118421, 10 to 160.027619, 20 fc rgb "#FF0000"
set object 25 rect from 0.084345, 10 to 112.839123, 20 fc rgb "#00FF00"
set object 26 rect from 0.086592, 10 to 113.809510, 20 fc rgb "#0000FF"
set object 27 rect from 0.087021, 10 to 114.819159, 20 fc rgb "#FFFF00"
set object 28 rect from 0.087843, 10 to 116.437797, 20 fc rgb "#FF00FF"
set object 29 rect from 0.089060, 10 to 118.134960, 20 fc rgb "#808080"
set object 30 rect from 0.090359, 10 to 118.953450, 20 fc rgb "#800080"
set object 31 rect from 0.091252, 10 to 119.769287, 20 fc rgb "#008080"
set object 32 rect from 0.091594, 10 to 154.252482, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.116723, 20 to 157.614101, 30 fc rgb "#FF0000"
set object 34 rect from 0.070261, 20 to 94.361301, 30 fc rgb "#00FF00"
set object 35 rect from 0.072439, 20 to 95.165391, 30 fc rgb "#0000FF"
set object 36 rect from 0.072781, 20 to 96.219570, 30 fc rgb "#FFFF00"
set object 37 rect from 0.073589, 20 to 97.713749, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074728, 20 to 99.241999, 30 fc rgb "#808080"
set object 39 rect from 0.075903, 20 to 100.074891, 30 fc rgb "#800080"
set object 40 rect from 0.076806, 20 to 100.968003, 30 fc rgb "#008080"
set object 41 rect from 0.077216, 20 to 152.191269, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.115154, 30 to 155.626260, 40 fc rgb "#FF0000"
set object 43 rect from 0.054709, 30 to 73.865470, 40 fc rgb "#00FF00"
set object 44 rect from 0.056861, 30 to 74.759869, 40 fc rgb "#0000FF"
set object 45 rect from 0.057202, 30 to 75.935855, 40 fc rgb "#FFFF00"
set object 46 rect from 0.058100, 30 to 77.436629, 40 fc rgb "#FF00FF"
set object 47 rect from 0.059247, 30 to 78.950478, 40 fc rgb "#808080"
set object 48 rect from 0.060427, 30 to 79.775486, 40 fc rgb "#800080"
set object 49 rect from 0.061298, 30 to 80.667271, 40 fc rgb "#008080"
set object 50 rect from 0.061717, 30 to 150.147032, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.113669, 40 to 153.816386, 50 fc rgb "#FF0000"
set object 52 rect from 0.039457, 40 to 54.419876, 50 fc rgb "#00FF00"
set object 53 rect from 0.041921, 40 to 55.223926, 50 fc rgb "#0000FF"
set object 54 rect from 0.042281, 40 to 56.335750, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043141, 40 to 57.925546, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044371, 40 to 59.510113, 50 fc rgb "#808080"
set object 57 rect from 0.045563, 40 to 60.360022, 50 fc rgb "#800080"
set object 58 rect from 0.046488, 40 to 61.450887, 50 fc rgb "#008080"
set object 59 rect from 0.047074, 40 to 148.146039, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.111932, 50 to 156.333405, 60 fc rgb "#FF0000"
set object 61 rect from 0.018027, 50 to 27.449568, 60 fc rgb "#00FF00"
set object 62 rect from 0.021400, 50 to 28.975204, 60 fc rgb "#0000FF"
set object 63 rect from 0.022286, 50 to 33.159255, 60 fc rgb "#FFFF00"
set object 64 rect from 0.025481, 50 to 35.505957, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027230, 50 to 37.357669, 60 fc rgb "#808080"
set object 66 rect from 0.028670, 50 to 38.364742, 60 fc rgb "#800080"
set object 67 rect from 0.029984, 50 to 40.067134, 60 fc rgb "#008080"
set object 68 rect from 0.030731, 50 to 145.338341, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
