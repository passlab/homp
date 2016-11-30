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

set object 15 rect from 0.125259, 0 to 160.421868, 10 fc rgb "#FF0000"
set object 16 rect from 0.104513, 0 to 133.502145, 10 fc rgb "#00FF00"
set object 17 rect from 0.106773, 0 to 134.278875, 10 fc rgb "#0000FF"
set object 18 rect from 0.107121, 0 to 135.162260, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107833, 0 to 136.452222, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108872, 0 to 137.001804, 10 fc rgb "#808080"
set object 21 rect from 0.109299, 0 to 137.679432, 10 fc rgb "#800080"
set object 22 rect from 0.110097, 0 to 138.335745, 10 fc rgb "#008080"
set object 23 rect from 0.110366, 0 to 156.467914, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.123671, 10 to 158.927421, 20 fc rgb "#FF0000"
set object 25 rect from 0.089426, 10 to 114.675967, 20 fc rgb "#00FF00"
set object 26 rect from 0.091748, 10 to 115.535568, 20 fc rgb "#0000FF"
set object 27 rect from 0.092186, 10 to 116.505489, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092992, 10 to 118.032621, 20 fc rgb "#FF00FF"
set object 29 rect from 0.094209, 10 to 118.643758, 20 fc rgb "#808080"
set object 30 rect from 0.094688, 10 to 119.409194, 20 fc rgb "#800080"
set object 31 rect from 0.095536, 10 to 120.172087, 20 fc rgb "#008080"
set object 32 rect from 0.095886, 10 to 154.356047, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121962, 20 to 156.560807, 30 fc rgb "#FF0000"
set object 34 rect from 0.076179, 20 to 97.781031, 30 fc rgb "#00FF00"
set object 35 rect from 0.078303, 20 to 98.648111, 30 fc rgb "#0000FF"
set object 36 rect from 0.078727, 20 to 99.654456, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079532, 20 to 101.012181, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080623, 20 to 101.579340, 30 fc rgb "#808080"
set object 39 rect from 0.081071, 20 to 102.224358, 30 fc rgb "#800080"
set object 40 rect from 0.081840, 20 to 102.978501, 30 fc rgb "#008080"
set object 41 rect from 0.082203, 20 to 152.390999, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.120355, 30 to 154.580651, 40 fc rgb "#FF0000"
set object 43 rect from 0.060821, 30 to 78.884622, 40 fc rgb "#00FF00"
set object 44 rect from 0.063228, 30 to 79.637493, 40 fc rgb "#0000FF"
set object 45 rect from 0.063583, 30 to 80.776896, 40 fc rgb "#FFFF00"
set object 46 rect from 0.064489, 30 to 82.122056, 40 fc rgb "#FF00FF"
set object 47 rect from 0.065566, 30 to 82.654061, 40 fc rgb "#808080"
set object 48 rect from 0.066003, 30 to 83.364300, 40 fc rgb "#800080"
set object 49 rect from 0.066807, 30 to 84.146040, 40 fc rgb "#008080"
set object 50 rect from 0.067174, 30 to 150.285415, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.118599, 40 to 152.565417, 50 fc rgb "#FF0000"
set object 52 rect from 0.044872, 40 to 59.073415, 50 fc rgb "#00FF00"
set object 53 rect from 0.047461, 40 to 59.951789, 50 fc rgb "#0000FF"
set object 54 rect from 0.047901, 40 to 61.083638, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048818, 40 to 62.596934, 50 fc rgb "#FF00FF"
set object 56 rect from 0.050022, 40 to 63.120188, 50 fc rgb "#808080"
set object 57 rect from 0.050417, 40 to 63.780240, 50 fc rgb "#800080"
set object 58 rect from 0.051234, 40 to 64.602145, 50 fc rgb "#008080"
set object 59 rect from 0.051624, 40 to 148.000401, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116780, 50 to 153.404824, 60 fc rgb "#FF0000"
set object 61 rect from 0.024263, 50 to 34.111186, 60 fc rgb "#00FF00"
set object 62 rect from 0.027714, 50 to 35.732408, 60 fc rgb "#0000FF"
set object 63 rect from 0.028629, 50 to 38.405151, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030779, 50 to 40.512007, 60 fc rgb "#FF00FF"
set object 65 rect from 0.032415, 50 to 41.344009, 60 fc rgb "#808080"
set object 66 rect from 0.033091, 50 to 42.223579, 60 fc rgb "#800080"
set object 67 rect from 0.034280, 50 to 43.869932, 60 fc rgb "#008080"
set object 68 rect from 0.035098, 50 to 145.292430, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
