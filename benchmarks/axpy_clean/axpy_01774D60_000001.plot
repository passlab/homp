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

set object 15 rect from 0.230399, 0 to 156.917006, 10 fc rgb "#FF0000"
set object 16 rect from 0.140815, 0 to 92.478534, 10 fc rgb "#00FF00"
set object 17 rect from 0.142942, 0 to 92.920904, 10 fc rgb "#0000FF"
set object 18 rect from 0.143380, 0 to 93.373655, 10 fc rgb "#FFFF00"
set object 19 rect from 0.144102, 0 to 94.083266, 10 fc rgb "#FF00FF"
set object 20 rect from 0.145186, 0 to 100.055291, 10 fc rgb "#808080"
set object 21 rect from 0.154395, 0 to 100.469776, 10 fc rgb "#800080"
set object 22 rect from 0.155274, 0 to 100.820691, 10 fc rgb "#008080"
set object 23 rect from 0.155557, 0 to 148.962745, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.226368, 10 to 154.499522, 20 fc rgb "#FF0000"
set object 25 rect from 0.082472, 10 to 54.778928, 20 fc rgb "#00FF00"
set object 26 rect from 0.084835, 10 to 55.282926, 20 fc rgb "#0000FF"
set object 27 rect from 0.085348, 10 to 55.773294, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086157, 10 to 56.583446, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087383, 10 to 62.592450, 20 fc rgb "#808080"
set object 30 rect from 0.096648, 10 to 63.001088, 20 fc rgb "#800080"
set object 31 rect from 0.097581, 10 to 63.434382, 20 fc rgb "#008080"
set object 32 rect from 0.097929, 10 to 146.195641, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.224245, 20 to 159.950712, 30 fc rgb "#FF0000"
set object 34 rect from 0.043757, 20 to 30.478288, 30 fc rgb "#00FF00"
set object 35 rect from 0.047623, 20 to 31.673735, 30 fc rgb "#0000FF"
set object 36 rect from 0.048963, 20 to 34.783963, 30 fc rgb "#FFFF00"
set object 37 rect from 0.053796, 20 to 39.121418, 30 fc rgb "#FF00FF"
set object 38 rect from 0.060479, 20 to 45.181660, 30 fc rgb "#808080"
set object 39 rect from 0.069897, 20 to 45.760893, 30 fc rgb "#800080"
set object 40 rect from 0.071130, 20 to 46.639148, 30 fc rgb "#008080"
set object 41 rect from 0.072041, 20 to 144.794580, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.233652, 30 to 163.641450, 40 fc rgb "#FF0000"
set object 43 rect from 0.195490, 30 to 127.734057, 40 fc rgb "#00FF00"
set object 44 rect from 0.197310, 30 to 128.215998, 40 fc rgb "#0000FF"
set object 45 rect from 0.197789, 30 to 130.333155, 40 fc rgb "#FFFF00"
set object 46 rect from 0.201067, 30 to 134.010293, 40 fc rgb "#FF00FF"
set object 47 rect from 0.206725, 30 to 139.875302, 40 fc rgb "#808080"
set object 48 rect from 0.215784, 30 to 140.324805, 40 fc rgb "#800080"
set object 49 rect from 0.216761, 30 to 140.741880, 40 fc rgb "#008080"
set object 50 rect from 0.217102, 30 to 151.186280, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.232082, 40 to 162.691864, 50 fc rgb "#FF0000"
set object 52 rect from 0.164329, 40 to 107.624924, 50 fc rgb "#00FF00"
set object 53 rect from 0.166313, 40 to 108.088055, 50 fc rgb "#0000FF"
set object 54 rect from 0.166756, 40 to 110.338181, 50 fc rgb "#FFFF00"
set object 55 rect from 0.170238, 40 to 113.984834, 50 fc rgb "#FF00FF"
set object 56 rect from 0.175863, 40 to 119.823242, 50 fc rgb "#808080"
set object 57 rect from 0.184877, 40 to 120.298697, 50 fc rgb "#800080"
set object 58 rect from 0.185868, 40 to 120.731992, 50 fc rgb "#008080"
set object 59 rect from 0.186279, 40 to 150.178296, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.228434, 50 to 160.580535, 60 fc rgb "#FF0000"
set object 61 rect from 0.108273, 50 to 71.290065, 60 fc rgb "#00FF00"
set object 62 rect from 0.110280, 50 to 71.738921, 60 fc rgb "#0000FF"
set object 63 rect from 0.110719, 50 to 73.856726, 60 fc rgb "#FFFF00"
set object 64 rect from 0.114000, 50 to 77.909436, 60 fc rgb "#FF00FF"
set object 65 rect from 0.120244, 50 to 83.720597, 60 fc rgb "#808080"
set object 66 rect from 0.129216, 50 to 84.209023, 60 fc rgb "#800080"
set object 67 rect from 0.130208, 50 to 84.708478, 60 fc rgb "#008080"
set object 68 rect from 0.130735, 50 to 147.684919, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
