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

set object 15 rect from 0.130067, 0 to 152.818554, 10 fc rgb "#FF0000"
set object 16 rect from 0.033466, 0 to 41.126621, 10 fc rgb "#00FF00"
set object 17 rect from 0.037106, 0 to 42.590567, 10 fc rgb "#0000FF"
set object 18 rect from 0.037973, 0 to 44.443847, 10 fc rgb "#FFFF00"
set object 19 rect from 0.039650, 0 to 46.239740, 10 fc rgb "#FF00FF"
set object 20 rect from 0.041225, 0 to 47.934361, 10 fc rgb "#808080"
set object 21 rect from 0.042759, 0 to 48.707405, 10 fc rgb "#800080"
set object 22 rect from 0.043829, 0 to 49.591848, 10 fc rgb "#008080"
set object 23 rect from 0.044223, 0 to 145.191635, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138307, 10 to 159.736567, 20 fc rgb "#FF0000"
set object 25 rect from 0.115803, 10 to 132.502235, 20 fc rgb "#00FF00"
set object 26 rect from 0.118106, 10 to 133.314662, 20 fc rgb "#0000FF"
set object 27 rect from 0.118589, 10 to 134.193480, 20 fc rgb "#FFFF00"
set object 28 rect from 0.119394, 10 to 135.427874, 20 fc rgb "#FF00FF"
set object 29 rect from 0.120493, 10 to 136.861439, 20 fc rgb "#808080"
set object 30 rect from 0.121747, 10 to 137.448817, 20 fc rgb "#800080"
set object 31 rect from 0.122551, 10 to 138.082330, 20 fc rgb "#008080"
set object 32 rect from 0.122830, 10 to 154.989153, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.135193, 20 to 156.261807, 30 fc rgb "#FF0000"
set object 34 rect from 0.086884, 20 to 99.684430, 30 fc rgb "#00FF00"
set object 35 rect from 0.088969, 20 to 100.396712, 30 fc rgb "#0000FF"
set object 36 rect from 0.089351, 20 to 101.370050, 30 fc rgb "#FFFF00"
set object 37 rect from 0.090200, 20 to 102.657331, 30 fc rgb "#FF00FF"
set object 38 rect from 0.091342, 20 to 103.952489, 30 fc rgb "#808080"
set object 39 rect from 0.092495, 20 to 104.644515, 30 fc rgb "#800080"
set object 40 rect from 0.093375, 20 to 105.344419, 30 fc rgb "#008080"
set object 41 rect from 0.093734, 20 to 151.588660, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.132025, 30 to 153.588220, 40 fc rgb "#FF0000"
set object 43 rect from 0.053772, 30 to 63.058794, 40 fc rgb "#00FF00"
set object 44 rect from 0.056408, 30 to 63.864470, 40 fc rgb "#0000FF"
set object 45 rect from 0.056866, 30 to 65.087612, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057985, 30 to 66.747350, 40 fc rgb "#FF00FF"
set object 47 rect from 0.059455, 30 to 68.137030, 40 fc rgb "#808080"
set object 48 rect from 0.060688, 30 to 68.962960, 40 fc rgb "#800080"
set object 49 rect from 0.061714, 30 to 70.267121, 40 fc rgb "#008080"
set object 50 rect from 0.062591, 30 to 147.758321, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.133691, 40 to 154.705589, 50 fc rgb "#FF0000"
set object 52 rect from 0.070593, 40 to 81.581471, 50 fc rgb "#00FF00"
set object 53 rect from 0.072859, 40 to 82.281373, 50 fc rgb "#0000FF"
set object 54 rect from 0.073234, 40 to 83.296345, 50 fc rgb "#FFFF00"
set object 55 rect from 0.074140, 40 to 84.645516, 50 fc rgb "#FF00FF"
set object 56 rect from 0.075341, 40 to 85.953052, 50 fc rgb "#808080"
set object 57 rect from 0.076515, 40 to 86.661957, 50 fc rgb "#800080"
set object 58 rect from 0.077425, 40 to 87.412496, 50 fc rgb "#008080"
set object 59 rect from 0.077803, 40 to 149.728627, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.136673, 50 to 157.991309, 60 fc rgb "#FF0000"
set object 61 rect from 0.101033, 50 to 115.464883, 60 fc rgb "#00FF00"
set object 62 rect from 0.102972, 50 to 116.243553, 60 fc rgb "#0000FF"
set object 63 rect from 0.103435, 50 to 117.221392, 60 fc rgb "#FFFF00"
set object 64 rect from 0.104287, 50 to 118.477167, 60 fc rgb "#FF00FF"
set object 65 rect from 0.105401, 50 to 119.781326, 60 fc rgb "#808080"
set object 66 rect from 0.106563, 50 to 120.469977, 60 fc rgb "#800080"
set object 67 rect from 0.107442, 50 to 121.162004, 60 fc rgb "#008080"
set object 68 rect from 0.107794, 50 to 153.214640, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
