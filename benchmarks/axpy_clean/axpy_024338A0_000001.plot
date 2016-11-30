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

set object 15 rect from 0.121187, 0 to 162.132114, 10 fc rgb "#FF0000"
set object 16 rect from 0.100707, 0 to 132.694006, 10 fc rgb "#00FF00"
set object 17 rect from 0.102674, 0 to 133.411246, 10 fc rgb "#0000FF"
set object 18 rect from 0.102979, 0 to 134.297093, 10 fc rgb "#FFFF00"
set object 19 rect from 0.103669, 0 to 135.614858, 10 fc rgb "#FF00FF"
set object 20 rect from 0.104681, 0 to 137.923532, 10 fc rgb "#808080"
set object 21 rect from 0.106465, 0 to 138.597962, 10 fc rgb "#800080"
set object 22 rect from 0.107240, 0 to 139.273708, 10 fc rgb "#008080"
set object 23 rect from 0.107504, 0 to 156.474594, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.119663, 10 to 160.828651, 20 fc rgb "#FF0000"
set object 25 rect from 0.085624, 10 to 113.458114, 20 fc rgb "#00FF00"
set object 26 rect from 0.087843, 10 to 114.319320, 20 fc rgb "#0000FF"
set object 27 rect from 0.088260, 10 to 115.315428, 20 fc rgb "#FFFF00"
set object 28 rect from 0.089099, 10 to 116.880925, 20 fc rgb "#FF00FF"
set object 29 rect from 0.090262, 10 to 119.350417, 20 fc rgb "#808080"
set object 30 rect from 0.092168, 10 to 120.184391, 20 fc rgb "#800080"
set object 31 rect from 0.093070, 10 to 121.002788, 20 fc rgb "#008080"
set object 32 rect from 0.093421, 10 to 154.315086, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.117911, 20 to 159.816927, 30 fc rgb "#FF0000"
set object 34 rect from 0.070242, 20 to 93.189820, 30 fc rgb "#00FF00"
set object 35 rect from 0.072218, 20 to 94.010826, 30 fc rgb "#0000FF"
set object 36 rect from 0.072603, 20 to 96.672273, 30 fc rgb "#FFFF00"
set object 37 rect from 0.074685, 20 to 98.316874, 30 fc rgb "#FF00FF"
set object 38 rect from 0.075925, 20 to 100.236440, 30 fc rgb "#808080"
set object 39 rect from 0.077414, 20 to 101.075613, 30 fc rgb "#800080"
set object 40 rect from 0.078309, 20 to 101.879747, 30 fc rgb "#008080"
set object 41 rect from 0.078692, 20 to 152.242471, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.116322, 30 to 157.688574, 40 fc rgb "#FF0000"
set object 43 rect from 0.054184, 30 to 72.476640, 40 fc rgb "#00FF00"
set object 44 rect from 0.056249, 30 to 73.300255, 40 fc rgb "#0000FF"
set object 45 rect from 0.056652, 30 to 75.890366, 40 fc rgb "#FFFF00"
set object 46 rect from 0.058645, 30 to 77.520705, 40 fc rgb "#FF00FF"
set object 47 rect from 0.059886, 30 to 79.467501, 40 fc rgb "#808080"
set object 48 rect from 0.061411, 30 to 80.304066, 40 fc rgb "#800080"
set object 49 rect from 0.062310, 30 to 81.096545, 40 fc rgb "#008080"
set object 50 rect from 0.062669, 30 to 150.063500, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.114587, 40 to 155.994669, 50 fc rgb "#FF0000"
set object 52 rect from 0.037707, 40 to 51.411981, 50 fc rgb "#00FF00"
set object 53 rect from 0.040045, 40 to 52.277091, 50 fc rgb "#0000FF"
set object 54 rect from 0.040433, 40 to 55.002104, 50 fc rgb "#FFFF00"
set object 55 rect from 0.042565, 40 to 56.675232, 50 fc rgb "#FF00FF"
set object 56 rect from 0.043830, 40 to 58.939801, 50 fc rgb "#808080"
set object 57 rect from 0.045575, 40 to 59.839932, 50 fc rgb "#800080"
set object 58 rect from 0.046549, 40 to 60.725779, 50 fc rgb "#008080"
set object 59 rect from 0.046958, 40 to 147.841720, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.112861, 50 to 156.233337, 60 fc rgb "#FF0000"
set object 61 rect from 0.016722, 50 to 25.137258, 60 fc rgb "#00FF00"
set object 62 rect from 0.019857, 50 to 26.522454, 60 fc rgb "#0000FF"
set object 63 rect from 0.020581, 50 to 30.397920, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023620, 50 to 32.711773, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025367, 50 to 34.993195, 60 fc rgb "#808080"
set object 66 rect from 0.027121, 50 to 36.227951, 60 fc rgb "#800080"
set object 67 rect from 0.028467, 50 to 37.877732, 60 fc rgb "#008080"
set object 68 rect from 0.029336, 50 to 145.195832, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
