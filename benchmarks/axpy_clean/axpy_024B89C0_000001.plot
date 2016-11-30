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

set object 15 rect from 0.102700, 0 to 163.069132, 10 fc rgb "#FF0000"
set object 16 rect from 0.083986, 0 to 131.979671, 10 fc rgb "#00FF00"
set object 17 rect from 0.085792, 0 to 132.839907, 10 fc rgb "#0000FF"
set object 18 rect from 0.086122, 0 to 133.792819, 10 fc rgb "#FFFF00"
set object 19 rect from 0.086741, 0 to 135.222970, 10 fc rgb "#FF00FF"
set object 20 rect from 0.087667, 0 to 136.753507, 10 fc rgb "#808080"
set object 21 rect from 0.088659, 0 to 137.477846, 10 fc rgb "#800080"
set object 22 rect from 0.089367, 0 to 138.222276, 10 fc rgb "#008080"
set object 23 rect from 0.089610, 0 to 157.873659, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.101286, 10 to 161.453628, 20 fc rgb "#FF0000"
set object 25 rect from 0.071148, 10 to 112.173842, 20 fc rgb "#00FF00"
set object 26 rect from 0.072966, 10 to 113.120563, 20 fc rgb "#0000FF"
set object 27 rect from 0.073354, 10 to 114.158420, 20 fc rgb "#FFFF00"
set object 28 rect from 0.074067, 10 to 115.915989, 20 fc rgb "#FF00FF"
set object 29 rect from 0.075194, 10 to 117.500586, 20 fc rgb "#808080"
set object 30 rect from 0.076219, 10 to 118.353112, 20 fc rgb "#800080"
set object 31 rect from 0.076981, 10 to 119.184028, 20 fc rgb "#008080"
set object 32 rect from 0.077284, 10 to 155.496763, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.099614, 20 to 158.421895, 30 fc rgb "#FF0000"
set object 34 rect from 0.059465, 20 to 93.730203, 30 fc rgb "#00FF00"
set object 35 rect from 0.061044, 20 to 94.598171, 30 fc rgb "#0000FF"
set object 36 rect from 0.061364, 20 to 95.700881, 30 fc rgb "#FFFF00"
set object 37 rect from 0.062076, 20 to 97.304004, 30 fc rgb "#FF00FF"
set object 38 rect from 0.063114, 20 to 98.587442, 30 fc rgb "#808080"
set object 39 rect from 0.063944, 20 to 99.330307, 30 fc rgb "#800080"
set object 40 rect from 0.064666, 20 to 100.192107, 30 fc rgb "#008080"
set object 41 rect from 0.064986, 20 to 153.172385, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.098245, 30 to 156.489835, 40 fc rgb "#FF0000"
set object 43 rect from 0.046828, 30 to 74.446374, 40 fc rgb "#00FF00"
set object 44 rect from 0.048555, 30 to 75.280374, 40 fc rgb "#0000FF"
set object 45 rect from 0.048853, 30 to 76.599322, 40 fc rgb "#FFFF00"
set object 46 rect from 0.049708, 30 to 78.185461, 40 fc rgb "#FF00FF"
set object 47 rect from 0.050734, 30 to 79.498218, 40 fc rgb "#808080"
set object 48 rect from 0.051597, 30 to 80.282807, 40 fc rgb "#800080"
set object 49 rect from 0.052331, 30 to 81.121433, 40 fc rgb "#008080"
set object 50 rect from 0.052638, 30 to 150.922158, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.096802, 40 to 154.330697, 50 fc rgb "#FF0000"
set object 52 rect from 0.033820, 40 to 54.861409, 50 fc rgb "#00FF00"
set object 53 rect from 0.035879, 40 to 55.778788, 50 fc rgb "#0000FF"
set object 54 rect from 0.036228, 40 to 56.909322, 50 fc rgb "#FFFF00"
set object 55 rect from 0.036978, 40 to 58.657616, 50 fc rgb "#FF00FF"
set object 56 rect from 0.038112, 40 to 59.967313, 50 fc rgb "#808080"
set object 57 rect from 0.038938, 40 to 60.819838, 50 fc rgb "#800080"
set object 58 rect from 0.039743, 40 to 61.826810, 50 fc rgb "#008080"
set object 59 rect from 0.040155, 40 to 148.659549, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.095176, 50 to 155.549304, 60 fc rgb "#FF0000"
set object 61 rect from 0.015626, 50 to 27.948091, 60 fc rgb "#00FF00"
set object 62 rect from 0.018505, 50 to 29.424569, 60 fc rgb "#0000FF"
set object 63 rect from 0.019177, 50 to 32.603015, 60 fc rgb "#FFFF00"
set object 64 rect from 0.021270, 50 to 35.166760, 60 fc rgb "#FF00FF"
set object 65 rect from 0.022892, 50 to 36.757548, 60 fc rgb "#808080"
set object 66 rect from 0.023940, 50 to 37.810846, 60 fc rgb "#800080"
set object 67 rect from 0.024986, 50 to 39.702771, 60 fc rgb "#008080"
set object 68 rect from 0.025833, 50 to 145.468745, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
