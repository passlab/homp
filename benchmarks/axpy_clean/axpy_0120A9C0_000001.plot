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

set object 15 rect from 0.140624, 0 to 159.034744, 10 fc rgb "#FF0000"
set object 16 rect from 0.119116, 0 to 133.529099, 10 fc rgb "#00FF00"
set object 17 rect from 0.121349, 0 to 134.203466, 10 fc rgb "#0000FF"
set object 18 rect from 0.121708, 0 to 135.035664, 10 fc rgb "#FFFF00"
set object 19 rect from 0.122491, 0 to 136.205597, 10 fc rgb "#FF00FF"
set object 20 rect from 0.123522, 0 to 137.596272, 10 fc rgb "#808080"
set object 21 rect from 0.124787, 0 to 138.180135, 10 fc rgb "#800080"
set object 22 rect from 0.125573, 0 to 138.782761, 10 fc rgb "#008080"
set object 23 rect from 0.125859, 0 to 154.622110, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.139072, 10 to 157.612061, 20 fc rgb "#FF0000"
set object 25 rect from 0.103254, 10 to 115.982310, 20 fc rgb "#00FF00"
set object 26 rect from 0.105428, 10 to 116.741663, 20 fc rgb "#0000FF"
set object 27 rect from 0.105888, 10 to 117.557305, 20 fc rgb "#FFFF00"
set object 28 rect from 0.106664, 10 to 118.849750, 20 fc rgb "#FF00FF"
set object 29 rect from 0.107830, 10 to 120.334241, 20 fc rgb "#808080"
set object 30 rect from 0.109177, 10 to 120.976600, 20 fc rgb "#800080"
set object 31 rect from 0.109988, 10 to 121.626685, 20 fc rgb "#008080"
set object 32 rect from 0.110318, 10 to 152.794365, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137354, 20 to 155.966428, 30 fc rgb "#FF0000"
set object 34 rect from 0.088075, 20 to 99.232400, 30 fc rgb "#00FF00"
set object 35 rect from 0.090261, 20 to 100.007205, 30 fc rgb "#0000FF"
set object 36 rect from 0.090723, 20 to 100.919973, 30 fc rgb "#FFFF00"
set object 37 rect from 0.091569, 20 to 102.409982, 30 fc rgb "#FF00FF"
set object 38 rect from 0.092901, 20 to 103.746576, 30 fc rgb "#808080"
set object 39 rect from 0.094128, 20 to 104.404387, 30 fc rgb "#800080"
set object 40 rect from 0.094979, 20 to 105.074340, 30 fc rgb "#008080"
set object 41 rect from 0.095336, 20 to 151.069266, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.135867, 30 to 154.324108, 40 fc rgb "#FF0000"
set object 43 rect from 0.071000, 30 to 80.457181, 40 fc rgb "#00FF00"
set object 44 rect from 0.073256, 30 to 81.220949, 40 fc rgb "#0000FF"
set object 45 rect from 0.073705, 30 to 82.260644, 40 fc rgb "#FFFF00"
set object 46 rect from 0.074660, 30 to 83.603860, 40 fc rgb "#FF00FF"
set object 47 rect from 0.075865, 30 to 84.904031, 40 fc rgb "#808080"
set object 48 rect from 0.077056, 30 to 85.635791, 40 fc rgb "#800080"
set object 49 rect from 0.077980, 30 to 86.333336, 40 fc rgb "#008080"
set object 50 rect from 0.078337, 30 to 149.356307, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.134289, 40 to 152.531681, 50 fc rgb "#FF0000"
set object 52 rect from 0.055076, 40 to 63.132238, 50 fc rgb "#00FF00"
set object 53 rect from 0.057576, 40 to 63.891591, 50 fc rgb "#0000FF"
set object 54 rect from 0.058019, 40 to 64.877204, 50 fc rgb "#FFFF00"
set object 55 rect from 0.058903, 40 to 66.244702, 50 fc rgb "#FF00FF"
set object 56 rect from 0.060144, 40 to 67.579088, 50 fc rgb "#808080"
set object 57 rect from 0.061360, 40 to 68.302018, 50 fc rgb "#800080"
set object 58 rect from 0.062319, 40 to 69.064682, 50 fc rgb "#008080"
set object 59 rect from 0.062715, 40 to 147.505384, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.132436, 50 to 154.343974, 60 fc rgb "#FF0000"
set object 61 rect from 0.032860, 50 to 39.275537, 60 fc rgb "#00FF00"
set object 62 rect from 0.036121, 50 to 41.224690, 60 fc rgb "#0000FF"
set object 63 rect from 0.037486, 50 to 44.142900, 60 fc rgb "#FFFF00"
set object 64 rect from 0.040206, 50 to 46.037971, 60 fc rgb "#FF00FF"
set object 65 rect from 0.041852, 50 to 47.764174, 60 fc rgb "#808080"
set object 66 rect from 0.043427, 50 to 48.605201, 60 fc rgb "#800080"
set object 67 rect from 0.044595, 50 to 49.850186, 60 fc rgb "#008080"
set object 68 rect from 0.045312, 50 to 145.155585, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
