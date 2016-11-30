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

set object 15 rect from 0.135986, 0 to 156.411879, 10 fc rgb "#FF0000"
set object 16 rect from 0.087457, 0 to 100.830423, 10 fc rgb "#00FF00"
set object 17 rect from 0.089760, 0 to 101.610824, 10 fc rgb "#0000FF"
set object 18 rect from 0.090215, 0 to 102.455486, 10 fc rgb "#FFFF00"
set object 19 rect from 0.090978, 0 to 103.632877, 10 fc rgb "#FF00FF"
set object 20 rect from 0.092015, 0 to 104.107638, 10 fc rgb "#808080"
set object 21 rect from 0.092435, 0 to 104.692906, 10 fc rgb "#800080"
set object 22 rect from 0.093209, 0 to 105.278241, 10 fc rgb "#008080"
set object 23 rect from 0.093494, 0 to 152.551614, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.133820, 10 to 154.421418, 20 fc rgb "#FF0000"
set object 25 rect from 0.071600, 10 to 83.028061, 20 fc rgb "#00FF00"
set object 26 rect from 0.074056, 10 to 83.999025, 20 fc rgb "#0000FF"
set object 27 rect from 0.074599, 10 to 84.849334, 20 fc rgb "#FFFF00"
set object 28 rect from 0.075397, 10 to 86.215070, 20 fc rgb "#FF00FF"
set object 29 rect from 0.076582, 10 to 86.821780, 20 fc rgb "#808080"
set object 30 rect from 0.077125, 10 to 87.505186, 20 fc rgb "#800080"
set object 31 rect from 0.077994, 10 to 88.182945, 20 fc rgb "#008080"
set object 32 rect from 0.078321, 10 to 150.076242, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.131793, 20 to 152.102866, 30 fc rgb "#FF0000"
set object 34 rect from 0.056291, 20 to 65.781659, 30 fc rgb "#00FF00"
set object 35 rect from 0.058699, 20 to 66.600442, 30 fc rgb "#0000FF"
set object 36 rect from 0.059170, 20 to 67.595067, 30 fc rgb "#FFFF00"
set object 37 rect from 0.060056, 20 to 68.916776, 30 fc rgb "#FF00FF"
set object 38 rect from 0.061241, 20 to 69.427633, 30 fc rgb "#808080"
set object 39 rect from 0.061716, 20 to 70.121189, 30 fc rgb "#800080"
set object 40 rect from 0.062592, 20 to 70.860990, 30 fc rgb "#008080"
set object 41 rect from 0.062984, 20 to 147.903083, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.129931, 30 to 153.926357, 40 fc rgb "#FF0000"
set object 43 rect from 0.032889, 30 to 42.262806, 40 fc rgb "#00FF00"
set object 44 rect from 0.038034, 30 to 43.909312, 40 fc rgb "#0000FF"
set object 45 rect from 0.039090, 30 to 47.157220, 40 fc rgb "#FFFF00"
set object 46 rect from 0.042036, 30 to 49.238963, 40 fc rgb "#FF00FF"
set object 47 rect from 0.043824, 30 to 50.076903, 40 fc rgb "#808080"
set object 48 rect from 0.044552, 30 to 50.918204, 40 fc rgb "#800080"
set object 49 rect from 0.045780, 30 to 52.341344, 40 fc rgb "#008080"
set object 50 rect from 0.046559, 30 to 145.357804, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.139138, 40 to 160.079160, 50 fc rgb "#FF0000"
set object 52 rect from 0.115674, 40 to 132.501682, 50 fc rgb "#00FF00"
set object 53 rect from 0.117857, 40 to 133.245987, 50 fc rgb "#0000FF"
set object 54 rect from 0.118267, 40 to 134.175209, 50 fc rgb "#FFFF00"
set object 55 rect from 0.119095, 40 to 135.373976, 50 fc rgb "#FF00FF"
set object 56 rect from 0.120156, 40 to 135.828504, 50 fc rgb "#808080"
set object 57 rect from 0.120560, 40 to 136.438575, 50 fc rgb "#800080"
set object 58 rect from 0.121390, 40 to 137.143424, 50 fc rgb "#008080"
set object 59 rect from 0.121732, 40 to 156.374640, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.137657, 50 to 158.559024, 60 fc rgb "#FF0000"
set object 61 rect from 0.100719, 50 to 115.925108, 60 fc rgb "#00FF00"
set object 62 rect from 0.103148, 50 to 116.697645, 60 fc rgb "#0000FF"
set object 63 rect from 0.103597, 50 to 117.700202, 60 fc rgb "#FFFF00"
set object 64 rect from 0.104493, 50 to 118.944072, 60 fc rgb "#FF00FF"
set object 65 rect from 0.105594, 50 to 119.427840, 60 fc rgb "#808080"
set object 66 rect from 0.106019, 50 to 120.057135, 60 fc rgb "#800080"
set object 67 rect from 0.106850, 50 to 120.794652, 60 fc rgb "#008080"
set object 68 rect from 0.107237, 50 to 154.584892, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
