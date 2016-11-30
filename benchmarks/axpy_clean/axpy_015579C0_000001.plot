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

set object 15 rect from 0.108052, 0 to 163.274833, 10 fc rgb "#FF0000"
set object 16 rect from 0.088515, 0 to 132.317374, 10 fc rgb "#00FF00"
set object 17 rect from 0.090522, 0 to 133.199432, 10 fc rgb "#0000FF"
set object 18 rect from 0.090874, 0 to 134.188652, 10 fc rgb "#FFFF00"
set object 19 rect from 0.091562, 0 to 135.697414, 10 fc rgb "#FF00FF"
set object 20 rect from 0.092591, 0 to 137.336783, 10 fc rgb "#808080"
set object 21 rect from 0.093696, 0 to 138.110256, 10 fc rgb "#800080"
set object 22 rect from 0.094483, 0 to 138.874894, 10 fc rgb "#008080"
set object 23 rect from 0.094746, 0 to 157.797512, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.106518, 10 to 161.667722, 20 fc rgb "#FF0000"
set object 25 rect from 0.073728, 10 to 110.635560, 20 fc rgb "#00FF00"
set object 26 rect from 0.075749, 10 to 111.574851, 20 fc rgb "#0000FF"
set object 27 rect from 0.076142, 10 to 112.734307, 20 fc rgb "#FFFF00"
set object 28 rect from 0.076981, 10 to 114.533676, 20 fc rgb "#FF00FF"
set object 29 rect from 0.078188, 10 to 116.230301, 20 fc rgb "#808080"
set object 30 rect from 0.079326, 10 to 117.146125, 20 fc rgb "#800080"
set object 31 rect from 0.080203, 10 to 118.006182, 20 fc rgb "#008080"
set object 32 rect from 0.080528, 10 to 155.405228, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.104825, 20 to 158.867410, 30 fc rgb "#FF0000"
set object 34 rect from 0.061247, 20 to 91.892013, 30 fc rgb "#00FF00"
set object 35 rect from 0.063008, 20 to 92.859188, 30 fc rgb "#0000FF"
set object 36 rect from 0.063390, 20 to 94.045063, 30 fc rgb "#FFFF00"
set object 37 rect from 0.064197, 20 to 95.771037, 30 fc rgb "#FF00FF"
set object 38 rect from 0.065376, 20 to 97.213751, 30 fc rgb "#808080"
set object 39 rect from 0.066355, 20 to 98.020970, 30 fc rgb "#800080"
set object 40 rect from 0.067188, 20 to 98.913306, 30 fc rgb "#008080"
set object 41 rect from 0.067516, 20 to 153.183178, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.103356, 30 to 156.872896, 40 fc rgb "#FF0000"
set object 43 rect from 0.047240, 30 to 71.676380, 40 fc rgb "#00FF00"
set object 44 rect from 0.049220, 30 to 72.603949, 40 fc rgb "#0000FF"
set object 45 rect from 0.049589, 30 to 73.996756, 40 fc rgb "#FFFF00"
set object 46 rect from 0.050540, 30 to 75.762358, 40 fc rgb "#FF00FF"
set object 47 rect from 0.051742, 30 to 77.125817, 40 fc rgb "#808080"
set object 48 rect from 0.052689, 30 to 77.981477, 40 fc rgb "#800080"
set object 49 rect from 0.053559, 30 to 78.916372, 40 fc rgb "#008080"
set object 50 rect from 0.053893, 30 to 150.930314, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.101841, 40 to 154.583378, 50 fc rgb "#FF0000"
set object 52 rect from 0.033471, 40 to 51.811540, 50 fc rgb "#00FF00"
set object 53 rect from 0.035665, 40 to 52.761110, 50 fc rgb "#0000FF"
set object 54 rect from 0.036070, 40 to 53.860402, 50 fc rgb "#FFFF00"
set object 55 rect from 0.036861, 40 to 55.793309, 50 fc rgb "#FF00FF"
set object 56 rect from 0.038158, 40 to 57.262442, 50 fc rgb "#808080"
set object 57 rect from 0.039150, 40 to 58.109289, 50 fc rgb "#800080"
set object 58 rect from 0.040009, 40 to 59.145464, 50 fc rgb "#008080"
set object 59 rect from 0.040461, 40 to 148.530681, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.100065, 50 to 155.183620, 60 fc rgb "#FF0000"
set object 61 rect from 0.015034, 50 to 25.745792, 60 fc rgb "#00FF00"
set object 62 rect from 0.018123, 50 to 27.688978, 60 fc rgb "#0000FF"
set object 63 rect from 0.019000, 50 to 30.565615, 60 fc rgb "#FFFF00"
set object 64 rect from 0.021004, 50 to 32.935877, 60 fc rgb "#FF00FF"
set object 65 rect from 0.022575, 50 to 34.569407, 60 fc rgb "#808080"
set object 66 rect from 0.023699, 50 to 35.593837, 60 fc rgb "#800080"
set object 67 rect from 0.024844, 50 to 37.388789, 60 fc rgb "#008080"
set object 68 rect from 0.025633, 50 to 145.476460, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
