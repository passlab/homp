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

set object 15 rect from 0.120816, 0 to 160.964742, 10 fc rgb "#FF0000"
set object 16 rect from 0.100090, 0 to 133.071847, 10 fc rgb "#00FF00"
set object 17 rect from 0.102395, 0 to 133.867326, 10 fc rgb "#0000FF"
set object 18 rect from 0.102768, 0 to 134.846672, 10 fc rgb "#FFFF00"
set object 19 rect from 0.103522, 0 to 136.197719, 10 fc rgb "#FF00FF"
set object 20 rect from 0.104565, 0 to 136.712829, 10 fc rgb "#808080"
set object 21 rect from 0.104948, 0 to 137.411807, 10 fc rgb "#800080"
set object 22 rect from 0.105752, 0 to 138.101652, 10 fc rgb "#008080"
set object 23 rect from 0.106034, 0 to 156.832125, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.119251, 10 to 159.422055, 20 fc rgb "#FF0000"
set object 25 rect from 0.085769, 10 to 114.345300, 20 fc rgb "#00FF00"
set object 26 rect from 0.088050, 10 to 115.266000, 20 fc rgb "#0000FF"
set object 27 rect from 0.088502, 10 to 116.314447, 20 fc rgb "#FFFF00"
set object 28 rect from 0.089357, 10 to 117.887177, 20 fc rgb "#FF00FF"
set object 29 rect from 0.090542, 10 to 118.500109, 20 fc rgb "#808080"
set object 30 rect from 0.091002, 10 to 119.307325, 20 fc rgb "#800080"
set object 31 rect from 0.091882, 10 to 120.088462, 20 fc rgb "#008080"
set object 32 rect from 0.092206, 10 to 154.680392, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.117520, 20 to 156.829559, 30 fc rgb "#FF0000"
set object 34 rect from 0.072848, 20 to 97.388408, 30 fc rgb "#00FF00"
set object 35 rect from 0.075037, 20 to 98.143468, 30 fc rgb "#0000FF"
set object 36 rect from 0.075374, 20 to 99.262377, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076233, 20 to 100.655125, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077298, 20 to 101.208050, 30 fc rgb "#808080"
set object 39 rect from 0.077729, 20 to 101.884875, 30 fc rgb "#800080"
set object 40 rect from 0.078502, 20 to 102.634688, 30 fc rgb "#008080"
set object 41 rect from 0.078835, 20 to 152.574286, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.115944, 30 to 155.187651, 40 fc rgb "#FF0000"
set object 43 rect from 0.056470, 30 to 76.099369, 40 fc rgb "#00FF00"
set object 44 rect from 0.058815, 30 to 77.065696, 40 fc rgb "#0000FF"
set object 45 rect from 0.059211, 30 to 78.416704, 40 fc rgb "#FFFF00"
set object 46 rect from 0.060249, 30 to 79.861608, 40 fc rgb "#FF00FF"
set object 47 rect from 0.061356, 30 to 80.389776, 40 fc rgb "#808080"
set object 48 rect from 0.061764, 30 to 81.146118, 40 fc rgb "#800080"
set object 49 rect from 0.062623, 30 to 81.927295, 40 fc rgb "#008080"
set object 50 rect from 0.062955, 30 to 150.470822, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.114352, 40 to 152.909453, 50 fc rgb "#FF0000"
set object 52 rect from 0.042148, 40 to 57.777090, 50 fc rgb "#00FF00"
set object 53 rect from 0.044682, 40 to 58.589513, 50 fc rgb "#0000FF"
set object 54 rect from 0.045038, 40 to 59.735822, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045926, 40 to 61.219863, 50 fc rgb "#FF00FF"
set object 56 rect from 0.047075, 40 to 61.823623, 50 fc rgb "#808080"
set object 57 rect from 0.047545, 40 to 62.552604, 50 fc rgb "#800080"
set object 58 rect from 0.048378, 40 to 63.362462, 50 fc rgb "#008080"
set object 59 rect from 0.048725, 40 to 148.315203, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.112562, 50 to 155.307665, 60 fc rgb "#FF0000"
set object 61 rect from 0.021382, 50 to 31.546896, 60 fc rgb "#00FF00"
set object 62 rect from 0.024701, 50 to 33.917708, 60 fc rgb "#0000FF"
set object 63 rect from 0.026159, 50 to 37.250922, 60 fc rgb "#FFFF00"
set object 64 rect from 0.028747, 50 to 39.444395, 60 fc rgb "#FF00FF"
set object 65 rect from 0.030387, 50 to 40.368982, 60 fc rgb "#808080"
set object 66 rect from 0.031111, 50 to 41.375728, 60 fc rgb "#800080"
set object 67 rect from 0.032330, 50 to 42.701979, 60 fc rgb "#008080"
set object 68 rect from 0.032879, 50 to 145.394070, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
