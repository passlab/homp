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

set object 15 rect from 0.108441, 0 to 162.631378, 10 fc rgb "#FF0000"
set object 16 rect from 0.089348, 0 to 132.394917, 10 fc rgb "#00FF00"
set object 17 rect from 0.091186, 0 to 133.298728, 10 fc rgb "#0000FF"
set object 18 rect from 0.091563, 0 to 134.253541, 10 fc rgb "#FFFF00"
set object 19 rect from 0.092216, 0 to 135.706919, 10 fc rgb "#FF00FF"
set object 20 rect from 0.093215, 0 to 137.233153, 10 fc rgb "#808080"
set object 21 rect from 0.094262, 0 to 138.007239, 10 fc rgb "#800080"
set object 22 rect from 0.095081, 0 to 138.841061, 10 fc rgb "#008080"
set object 23 rect from 0.095365, 0 to 157.310653, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.106961, 10 to 161.045366, 20 fc rgb "#FF0000"
set object 25 rect from 0.075957, 10 to 113.111922, 20 fc rgb "#00FF00"
set object 26 rect from 0.077967, 10 to 114.063825, 20 fc rgb "#0000FF"
set object 27 rect from 0.078369, 10 to 115.104658, 20 fc rgb "#FFFF00"
set object 28 rect from 0.079110, 10 to 116.888949, 20 fc rgb "#FF00FF"
set object 29 rect from 0.080331, 10 to 118.496814, 20 fc rgb "#808080"
set object 30 rect from 0.081421, 10 to 119.402101, 20 fc rgb "#800080"
set object 31 rect from 0.082296, 10 to 120.298613, 20 fc rgb "#008080"
set object 32 rect from 0.082649, 10 to 155.001558, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.105308, 20 to 158.156075, 30 fc rgb "#FF0000"
set object 34 rect from 0.063439, 20 to 94.702109, 30 fc rgb "#00FF00"
set object 35 rect from 0.065346, 20 to 95.594276, 30 fc rgb "#0000FF"
set object 36 rect from 0.065699, 20 to 96.712352, 30 fc rgb "#FFFF00"
set object 37 rect from 0.066466, 20 to 98.327516, 30 fc rgb "#FF00FF"
set object 38 rect from 0.067571, 20 to 99.652603, 30 fc rgb "#808080"
set object 39 rect from 0.068481, 20 to 100.433945, 30 fc rgb "#800080"
set object 40 rect from 0.069273, 20 to 101.353786, 30 fc rgb "#008080"
set object 41 rect from 0.069666, 20 to 152.807676, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.103829, 30 to 156.243580, 40 fc rgb "#FF0000"
set object 43 rect from 0.049074, 30 to 73.767500, 40 fc rgb "#00FF00"
set object 44 rect from 0.050975, 30 to 74.624651, 40 fc rgb "#0000FF"
set object 45 rect from 0.051313, 30 to 75.888526, 40 fc rgb "#FFFF00"
set object 46 rect from 0.052179, 30 to 77.588232, 40 fc rgb "#FF00FF"
set object 47 rect from 0.053343, 30 to 78.927916, 40 fc rgb "#808080"
set object 48 rect from 0.054274, 30 to 79.732587, 40 fc rgb "#800080"
set object 49 rect from 0.055111, 30 to 80.640742, 40 fc rgb "#008080"
set object 50 rect from 0.055441, 30 to 150.586077, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.102343, 40 to 154.352894, 50 fc rgb "#FF0000"
set object 52 rect from 0.034839, 40 to 53.560319, 50 fc rgb "#00FF00"
set object 53 rect from 0.037129, 40 to 54.491803, 50 fc rgb "#0000FF"
set object 54 rect from 0.037501, 40 to 55.792084, 50 fc rgb "#FFFF00"
set object 55 rect from 0.038395, 40 to 57.557434, 50 fc rgb "#FF00FF"
set object 56 rect from 0.039628, 40 to 58.961242, 50 fc rgb "#808080"
set object 57 rect from 0.040579, 40 to 59.873784, 50 fc rgb "#800080"
set object 58 rect from 0.041502, 40 to 60.892721, 50 fc rgb "#008080"
set object 59 rect from 0.041911, 40 to 148.329505, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.100677, 50 to 155.017545, 60 fc rgb "#FF0000"
set object 61 rect from 0.016104, 50 to 27.850122, 60 fc rgb "#00FF00"
set object 62 rect from 0.019568, 50 to 29.460941, 60 fc rgb "#0000FF"
set object 63 rect from 0.020343, 50 to 32.475524, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022433, 50 to 34.772933, 60 fc rgb "#FF00FF"
set object 65 rect from 0.023985, 50 to 36.312330, 60 fc rgb "#808080"
set object 66 rect from 0.025065, 50 to 37.426019, 60 fc rgb "#800080"
set object 67 rect from 0.026242, 50 to 39.245282, 60 fc rgb "#008080"
set object 68 rect from 0.027052, 50 to 145.422750, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
