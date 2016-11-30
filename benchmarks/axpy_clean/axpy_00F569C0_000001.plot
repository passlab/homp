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

set object 15 rect from 0.125547, 0 to 161.265451, 10 fc rgb "#FF0000"
set object 16 rect from 0.104368, 0 to 132.686746, 10 fc rgb "#00FF00"
set object 17 rect from 0.106512, 0 to 133.437956, 10 fc rgb "#0000FF"
set object 18 rect from 0.106868, 0 to 134.474175, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107706, 0 to 135.766608, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108741, 0 to 137.295289, 10 fc rgb "#808080"
set object 21 rect from 0.109956, 0 to 137.925283, 10 fc rgb "#800080"
set object 22 rect from 0.110712, 0 to 138.561498, 10 fc rgb "#008080"
set object 23 rect from 0.110971, 0 to 156.287003, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.123966, 10 to 159.785495, 20 fc rgb "#FF0000"
set object 25 rect from 0.089574, 10 to 114.253802, 20 fc rgb "#00FF00"
set object 26 rect from 0.091832, 10 to 115.162510, 20 fc rgb "#0000FF"
set object 27 rect from 0.092247, 10 to 116.132459, 20 fc rgb "#FFFF00"
set object 28 rect from 0.093053, 10 to 117.673657, 20 fc rgb "#FF00FF"
set object 29 rect from 0.094272, 10 to 119.354845, 20 fc rgb "#808080"
set object 30 rect from 0.095624, 10 to 120.107284, 20 fc rgb "#800080"
set object 31 rect from 0.096464, 10 to 120.847281, 20 fc rgb "#008080"
set object 32 rect from 0.096798, 10 to 154.133319, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.122234, 20 to 157.404429, 30 fc rgb "#FF0000"
set object 34 rect from 0.075716, 20 to 96.859574, 30 fc rgb "#00FF00"
set object 35 rect from 0.077837, 20 to 97.647029, 30 fc rgb "#0000FF"
set object 36 rect from 0.078235, 20 to 98.759464, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079125, 20 to 100.245678, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080312, 20 to 101.679332, 30 fc rgb "#808080"
set object 39 rect from 0.081460, 20 to 102.338083, 30 fc rgb "#800080"
set object 40 rect from 0.082256, 20 to 103.148038, 30 fc rgb "#008080"
set object 41 rect from 0.082639, 20 to 152.193421, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.120712, 30 to 155.549502, 40 fc rgb "#FF0000"
set object 43 rect from 0.060169, 30 to 77.305403, 40 fc rgb "#00FF00"
set object 44 rect from 0.062309, 30 to 78.264102, 40 fc rgb "#0000FF"
set object 45 rect from 0.062742, 30 to 79.455324, 40 fc rgb "#FFFF00"
set object 46 rect from 0.063684, 30 to 80.829003, 40 fc rgb "#FF00FF"
set object 47 rect from 0.064776, 30 to 82.292680, 40 fc rgb "#808080"
set object 48 rect from 0.065964, 30 to 83.008911, 40 fc rgb "#800080"
set object 49 rect from 0.066806, 30 to 83.793870, 40 fc rgb "#008080"
set object 50 rect from 0.067152, 30 to 150.174736, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.119068, 40 to 153.687123, 50 fc rgb "#FF0000"
set object 52 rect from 0.045718, 40 to 59.553674, 50 fc rgb "#00FF00"
set object 53 rect from 0.048087, 40 to 60.486148, 50 fc rgb "#0000FF"
set object 54 rect from 0.048522, 40 to 61.673570, 50 fc rgb "#FFFF00"
set object 55 rect from 0.049454, 40 to 63.171035, 50 fc rgb "#FF00FF"
set object 56 rect from 0.050668, 40 to 64.662204, 50 fc rgb "#808080"
set object 57 rect from 0.051857, 40 to 65.427159, 50 fc rgb "#800080"
set object 58 rect from 0.052741, 40 to 66.273397, 50 fc rgb "#008080"
set object 59 rect from 0.053150, 40 to 147.893579, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.117133, 50 to 153.997054, 60 fc rgb "#FF0000"
set object 61 rect from 0.026005, 50 to 36.333418, 60 fc rgb "#00FF00"
set object 62 rect from 0.029707, 50 to 37.963348, 60 fc rgb "#0000FF"
set object 63 rect from 0.030512, 50 to 40.360766, 60 fc rgb "#FFFF00"
set object 64 rect from 0.032446, 50 to 42.531920, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034148, 50 to 44.219329, 60 fc rgb "#808080"
set object 66 rect from 0.035523, 50 to 45.245528, 60 fc rgb "#800080"
set object 67 rect from 0.036821, 50 to 46.811721, 60 fc rgb "#008080"
set object 68 rect from 0.037586, 50 to 145.344958, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
