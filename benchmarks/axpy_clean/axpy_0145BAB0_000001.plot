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

set object 15 rect from 0.112853, 0 to 162.686298, 10 fc rgb "#FF0000"
set object 16 rect from 0.092170, 0 to 132.394825, 10 fc rgb "#00FF00"
set object 17 rect from 0.094390, 0 to 133.270681, 10 fc rgb "#0000FF"
set object 18 rect from 0.094766, 0 to 134.356269, 10 fc rgb "#FFFF00"
set object 19 rect from 0.095536, 0 to 135.841766, 10 fc rgb "#FF00FF"
set object 20 rect from 0.096601, 0 to 136.458457, 10 fc rgb "#808080"
set object 21 rect from 0.097027, 0 to 137.189288, 10 fc rgb "#800080"
set object 22 rect from 0.097808, 0 to 137.956628, 10 fc rgb "#008080"
set object 23 rect from 0.098094, 0 to 158.105846, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.111208, 10 to 160.724686, 20 fc rgb "#FF0000"
set object 25 rect from 0.077944, 10 to 112.393484, 20 fc rgb "#00FF00"
set object 26 rect from 0.080174, 10 to 113.296029, 20 fc rgb "#0000FF"
set object 27 rect from 0.080579, 10 to 114.367517, 20 fc rgb "#FFFF00"
set object 28 rect from 0.081376, 10 to 116.051584, 20 fc rgb "#FF00FF"
set object 29 rect from 0.082557, 10 to 116.751363, 20 fc rgb "#808080"
set object 30 rect from 0.083068, 10 to 117.648284, 20 fc rgb "#800080"
set object 31 rect from 0.083927, 10 to 118.486122, 20 fc rgb "#008080"
set object 32 rect from 0.084270, 10 to 155.644621, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.109443, 20 to 157.987678, 30 fc rgb "#FF0000"
set object 34 rect from 0.065458, 20 to 94.680145, 30 fc rgb "#00FF00"
set object 35 rect from 0.067595, 20 to 95.501113, 30 fc rgb "#0000FF"
set object 36 rect from 0.067940, 20 to 96.686658, 30 fc rgb "#FFFF00"
set object 37 rect from 0.068828, 20 to 98.294688, 30 fc rgb "#FF00FF"
set object 38 rect from 0.069922, 20 to 98.864884, 30 fc rgb "#808080"
set object 39 rect from 0.070333, 20 to 99.619634, 30 fc rgb "#800080"
set object 40 rect from 0.071130, 20 to 100.502456, 30 fc rgb "#008080"
set object 41 rect from 0.071503, 20 to 153.401520, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.107879, 30 to 156.006427, 40 fc rgb "#FF0000"
set object 43 rect from 0.051349, 30 to 75.081484, 40 fc rgb "#00FF00"
set object 44 rect from 0.053705, 30 to 75.971439, 40 fc rgb "#0000FF"
set object 45 rect from 0.054075, 30 to 77.263990, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054996, 30 to 78.793129, 40 fc rgb "#FF00FF"
set object 47 rect from 0.056077, 30 to 79.430970, 40 fc rgb "#808080"
set object 48 rect from 0.056543, 30 to 80.257477, 40 fc rgb "#800080"
set object 49 rect from 0.057388, 30 to 81.134760, 40 fc rgb "#008080"
set object 50 rect from 0.057759, 30 to 151.145830, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.106319, 40 to 153.929585, 50 fc rgb "#FF0000"
set object 52 rect from 0.037683, 40 to 55.896831, 50 fc rgb "#00FF00"
set object 53 rect from 0.040091, 40 to 56.833198, 50 fc rgb "#0000FF"
set object 54 rect from 0.040477, 40 to 58.029989, 50 fc rgb "#FFFF00"
set object 55 rect from 0.041339, 40 to 59.664791, 50 fc rgb "#FF00FF"
set object 56 rect from 0.042503, 40 to 60.418030, 50 fc rgb "#808080"
set object 57 rect from 0.043027, 40 to 61.202323, 50 fc rgb "#800080"
set object 58 rect from 0.043879, 40 to 62.180988, 50 fc rgb "#008080"
set object 59 rect from 0.044304, 40 to 148.870417, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.104466, 50 to 156.699240, 60 fc rgb "#FF0000"
set object 61 rect from 0.017395, 50 to 28.490496, 60 fc rgb "#00FF00"
set object 62 rect from 0.020678, 50 to 30.275947, 60 fc rgb "#0000FF"
set object 63 rect from 0.021636, 50 to 35.030881, 60 fc rgb "#FFFF00"
set object 64 rect from 0.025058, 50 to 37.404907, 60 fc rgb "#FF00FF"
set object 65 rect from 0.026700, 50 to 38.382062, 60 fc rgb "#808080"
set object 66 rect from 0.027395, 50 to 39.356448, 60 fc rgb "#800080"
set object 67 rect from 0.028583, 50 to 40.905309, 60 fc rgb "#008080"
set object 68 rect from 0.029193, 50 to 145.543239, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
