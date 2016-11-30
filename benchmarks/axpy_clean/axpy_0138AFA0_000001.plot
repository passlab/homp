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

set object 15 rect from 0.191319, 0 to 155.671103, 10 fc rgb "#FF0000"
set object 16 rect from 0.100862, 0 to 79.561114, 10 fc rgb "#00FF00"
set object 17 rect from 0.102848, 0 to 80.059342, 10 fc rgb "#0000FF"
set object 18 rect from 0.103271, 0 to 80.569967, 10 fc rgb "#FFFF00"
set object 19 rect from 0.103964, 0 to 81.437599, 10 fc rgb "#FF00FF"
set object 20 rect from 0.105057, 0 to 86.861445, 10 fc rgb "#808080"
set object 21 rect from 0.112048, 0 to 87.311567, 10 fc rgb "#800080"
set object 22 rect from 0.112865, 0 to 87.749268, 10 fc rgb "#008080"
set object 23 rect from 0.113196, 0 to 148.019200, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.192848, 10 to 156.371888, 20 fc rgb "#FF0000"
set object 25 rect from 0.122254, 10 to 95.904049, 20 fc rgb "#00FF00"
set object 26 rect from 0.123935, 10 to 96.326995, 20 fc rgb "#0000FF"
set object 27 rect from 0.124257, 10 to 96.815902, 20 fc rgb "#FFFF00"
set object 28 rect from 0.124860, 10 to 97.518237, 20 fc rgb "#FF00FF"
set object 29 rect from 0.125768, 10 to 102.795404, 20 fc rgb "#808080"
set object 30 rect from 0.132566, 10 to 103.162472, 20 fc rgb "#800080"
set object 31 rect from 0.133278, 10 to 103.569135, 20 fc rgb "#008080"
set object 32 rect from 0.133567, 10 to 149.242266, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.195713, 20 to 162.386314, 30 fc rgb "#FF0000"
set object 34 rect from 0.164972, 20 to 128.974055, 30 fc rgb "#00FF00"
set object 35 rect from 0.166516, 20 to 129.401650, 30 fc rgb "#0000FF"
set object 36 rect from 0.166852, 20 to 131.651425, 30 fc rgb "#FFFF00"
set object 37 rect from 0.169751, 20 to 134.303978, 30 fc rgb "#FF00FF"
set object 38 rect from 0.173172, 20 to 139.498901, 30 fc rgb "#808080"
set object 39 rect from 0.179867, 20 to 139.990144, 30 fc rgb "#800080"
set object 40 rect from 0.180732, 20 to 140.416976, 30 fc rgb "#008080"
set object 41 rect from 0.181049, 20 to 151.472637, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.189702, 30 to 159.438830, 40 fc rgb "#FF0000"
set object 43 rect from 0.074041, 30 to 58.814079, 40 fc rgb "#00FF00"
set object 44 rect from 0.076110, 30 to 60.259848, 40 fc rgb "#0000FF"
set object 45 rect from 0.077766, 30 to 62.735469, 40 fc rgb "#FFFF00"
set object 46 rect from 0.080977, 30 to 65.728724, 40 fc rgb "#FF00FF"
set object 47 rect from 0.084816, 30 to 71.053998, 40 fc rgb "#808080"
set object 48 rect from 0.091681, 30 to 71.570844, 40 fc rgb "#800080"
set object 49 rect from 0.092575, 30 to 72.048118, 40 fc rgb "#008080"
set object 50 rect from 0.092959, 30 to 146.715415, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.194264, 40 to 161.236170, 50 fc rgb "#FF0000"
set object 52 rect from 0.140518, 40 to 110.015804, 50 fc rgb "#00FF00"
set object 53 rect from 0.142092, 40 to 110.455842, 50 fc rgb "#0000FF"
set object 54 rect from 0.142439, 40 to 112.727334, 50 fc rgb "#FFFF00"
set object 55 rect from 0.145378, 40 to 115.339551, 50 fc rgb "#FF00FF"
set object 56 rect from 0.148733, 40 to 120.538336, 50 fc rgb "#808080"
set object 57 rect from 0.155434, 40 to 121.014061, 50 fc rgb "#800080"
set object 58 rect from 0.156303, 40 to 121.502182, 50 fc rgb "#008080"
set object 59 rect from 0.156676, 40 to 150.384616, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.187890, 50 to 161.122842, 60 fc rgb "#FF0000"
set object 61 rect from 0.038627, 50 to 32.899274, 60 fc rgb "#00FF00"
set object 62 rect from 0.042810, 50 to 35.957704, 60 fc rgb "#0000FF"
set object 63 rect from 0.046452, 50 to 39.441416, 60 fc rgb "#FFFF00"
set object 64 rect from 0.050972, 50 to 42.800189, 60 fc rgb "#FF00FF"
set object 65 rect from 0.055268, 50 to 48.248875, 60 fc rgb "#808080"
set object 66 rect from 0.062320, 50 to 48.854187, 60 fc rgb "#800080"
set object 67 rect from 0.063509, 50 to 49.864613, 60 fc rgb "#008080"
set object 68 rect from 0.064466, 50 to 144.899457, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
