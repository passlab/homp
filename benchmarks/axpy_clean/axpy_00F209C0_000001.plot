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

set object 15 rect from 0.122343, 0 to 161.803028, 10 fc rgb "#FF0000"
set object 16 rect from 0.101032, 0 to 132.342207, 10 fc rgb "#00FF00"
set object 17 rect from 0.103307, 0 to 133.216414, 10 fc rgb "#0000FF"
set object 18 rect from 0.103736, 0 to 134.179317, 10 fc rgb "#FFFF00"
set object 19 rect from 0.104503, 0 to 135.525360, 10 fc rgb "#FF00FF"
set object 20 rect from 0.105535, 0 to 137.136189, 10 fc rgb "#808080"
set object 21 rect from 0.106788, 0 to 137.802122, 10 fc rgb "#800080"
set object 22 rect from 0.107584, 0 to 138.540085, 10 fc rgb "#008080"
set object 23 rect from 0.107884, 0 to 156.568016, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.120745, 10 to 160.259019, 20 fc rgb "#FF0000"
set object 25 rect from 0.085994, 10 to 113.031222, 20 fc rgb "#00FF00"
set object 26 rect from 0.088277, 10 to 113.950409, 20 fc rgb "#0000FF"
set object 27 rect from 0.088754, 10 to 114.968600, 20 fc rgb "#FFFF00"
set object 28 rect from 0.089590, 10 to 116.569161, 20 fc rgb "#FF00FF"
set object 29 rect from 0.090823, 10 to 118.302173, 20 fc rgb "#808080"
set object 30 rect from 0.092170, 10 to 119.110825, 20 fc rgb "#800080"
set object 31 rect from 0.093056, 10 to 119.916872, 20 fc rgb "#008080"
set object 32 rect from 0.093395, 10 to 154.430064, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.119026, 20 to 157.988617, 30 fc rgb "#FF0000"
set object 34 rect from 0.071598, 20 to 94.316745, 30 fc rgb "#00FF00"
set object 35 rect from 0.073733, 20 to 95.112524, 30 fc rgb "#0000FF"
set object 36 rect from 0.074100, 20 to 96.260560, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075004, 20 to 97.804570, 30 fc rgb "#FF00FF"
set object 38 rect from 0.076202, 20 to 99.434709, 30 fc rgb "#808080"
set object 39 rect from 0.077466, 20 to 100.244625, 30 fc rgb "#800080"
set object 40 rect from 0.078353, 20 to 101.109829, 30 fc rgb "#008080"
set object 41 rect from 0.078779, 20 to 152.411691, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.117497, 30 to 156.006258, 40 fc rgb "#FF0000"
set object 43 rect from 0.056510, 30 to 74.956910, 40 fc rgb "#00FF00"
set object 44 rect from 0.058764, 30 to 75.838818, 40 fc rgb "#0000FF"
set object 45 rect from 0.059107, 30 to 77.022869, 40 fc rgb "#FFFF00"
set object 46 rect from 0.060040, 30 to 78.610557, 40 fc rgb "#FF00FF"
set object 47 rect from 0.061268, 30 to 80.168704, 40 fc rgb "#808080"
set object 48 rect from 0.062480, 30 to 80.977357, 40 fc rgb "#800080"
set object 49 rect from 0.063378, 30 to 81.807848, 40 fc rgb "#008080"
set object 50 rect from 0.063754, 30 to 150.335463, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.115887, 40 to 153.904283, 50 fc rgb "#FF0000"
set object 52 rect from 0.041700, 40 to 56.208985, 50 fc rgb "#00FF00"
set object 53 rect from 0.044195, 40 to 57.135912, 50 fc rgb "#0000FF"
set object 54 rect from 0.044575, 40 to 58.247971, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045436, 40 to 59.799682, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046656, 40 to 61.443960, 50 fc rgb "#808080"
set object 57 rect from 0.047924, 40 to 62.269317, 50 fc rgb "#800080"
set object 58 rect from 0.048832, 40 to 63.175669, 50 fc rgb "#008080"
set object 59 rect from 0.049281, 40 to 148.081804, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.113990, 50 to 154.508493, 60 fc rgb "#FF0000"
set object 61 rect from 0.021024, 50 to 30.719432, 60 fc rgb "#00FF00"
set object 62 rect from 0.024507, 50 to 32.524397, 60 fc rgb "#0000FF"
set object 63 rect from 0.025432, 50 to 34.857749, 60 fc rgb "#FFFF00"
set object 64 rect from 0.027279, 50 to 37.162825, 60 fc rgb "#FF00FF"
set object 65 rect from 0.029047, 50 to 39.066832, 60 fc rgb "#808080"
set object 66 rect from 0.030543, 50 to 40.076019, 60 fc rgb "#800080"
set object 67 rect from 0.031771, 50 to 41.789721, 60 fc rgb "#008080"
set object 68 rect from 0.032661, 50 to 145.397459, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
