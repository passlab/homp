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

set object 15 rect from 0.123970, 0 to 161.890299, 10 fc rgb "#FF0000"
set object 16 rect from 0.103443, 0 to 132.836464, 10 fc rgb "#00FF00"
set object 17 rect from 0.105240, 0 to 133.620472, 10 fc rgb "#0000FF"
set object 18 rect from 0.105618, 0 to 134.470338, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106307, 0 to 135.771108, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107326, 0 to 138.014196, 10 fc rgb "#808080"
set object 21 rect from 0.109100, 0 to 138.677887, 10 fc rgb "#800080"
set object 22 rect from 0.109901, 0 to 139.370699, 10 fc rgb "#008080"
set object 23 rect from 0.110160, 0 to 156.295852, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.122420, 10 to 160.674383, 20 fc rgb "#FF0000"
set object 25 rect from 0.087110, 10 to 112.353515, 20 fc rgb "#00FF00"
set object 26 rect from 0.089070, 10 to 113.189453, 20 fc rgb "#0000FF"
set object 27 rect from 0.089487, 10 to 114.221702, 20 fc rgb "#FFFF00"
set object 28 rect from 0.090348, 10 to 115.789718, 20 fc rgb "#FF00FF"
set object 29 rect from 0.091581, 10 to 118.219001, 20 fc rgb "#808080"
set object 30 rect from 0.093476, 10 to 119.011870, 20 fc rgb "#800080"
set object 31 rect from 0.094375, 10 to 119.794605, 20 fc rgb "#008080"
set object 32 rect from 0.094735, 10 to 154.193352, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.120602, 20 to 159.371075, 30 fc rgb "#FF0000"
set object 34 rect from 0.071824, 20 to 92.762227, 30 fc rgb "#00FF00"
set object 35 rect from 0.073605, 20 to 93.634893, 30 fc rgb "#0000FF"
set object 36 rect from 0.074049, 20 to 96.038848, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075972, 20 to 97.624586, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077202, 20 to 99.547242, 30 fc rgb "#808080"
set object 39 rect from 0.078729, 20 to 100.360381, 30 fc rgb "#800080"
set object 40 rect from 0.079629, 20 to 101.131716, 30 fc rgb "#008080"
set object 41 rect from 0.079971, 20 to 152.225098, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119165, 30 to 157.860056, 40 fc rgb "#FF0000"
set object 43 rect from 0.055200, 30 to 71.867651, 40 fc rgb "#00FF00"
set object 44 rect from 0.057129, 30 to 72.783376, 40 fc rgb "#0000FF"
set object 45 rect from 0.057585, 30 to 75.336779, 40 fc rgb "#FFFF00"
set object 46 rect from 0.059622, 30 to 76.937720, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060882, 30 to 78.968030, 40 fc rgb "#808080"
set object 48 rect from 0.062483, 30 to 79.786236, 40 fc rgb "#800080"
set object 49 rect from 0.063401, 30 to 80.653835, 40 fc rgb "#008080"
set object 50 rect from 0.063827, 30 to 150.144133, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.117555, 40 to 155.862700, 50 fc rgb "#FF0000"
set object 52 rect from 0.038682, 40 to 51.308699, 50 fc rgb "#00FF00"
set object 53 rect from 0.040881, 40 to 52.158575, 50 fc rgb "#0000FF"
set object 54 rect from 0.041302, 40 to 54.794304, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043401, 40 to 56.419308, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044666, 40 to 58.334358, 50 fc rgb "#808080"
set object 57 rect from 0.046212, 40 to 59.274156, 50 fc rgb "#800080"
set object 58 rect from 0.047202, 40 to 60.116427, 50 fc rgb "#008080"
set object 59 rect from 0.047634, 40 to 148.118891, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.115749, 50 to 157.078596, 60 fc rgb "#FF0000"
set object 61 rect from 0.016820, 50 to 24.615768, 60 fc rgb "#00FF00"
set object 62 rect from 0.020106, 50 to 26.318031, 60 fc rgb "#0000FF"
set object 63 rect from 0.020924, 50 to 30.539507, 60 fc rgb "#FFFF00"
set object 64 rect from 0.024277, 50 to 33.037187, 60 fc rgb "#FF00FF"
set object 65 rect from 0.026248, 50 to 35.425929, 60 fc rgb "#808080"
set object 66 rect from 0.028146, 50 to 36.593708, 60 fc rgb "#800080"
set object 67 rect from 0.029438, 50 to 38.064197, 60 fc rgb "#008080"
set object 68 rect from 0.030219, 50 to 145.299505, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
