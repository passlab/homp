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

set object 15 rect from 0.135216, 0 to 157.117146, 10 fc rgb "#FF0000"
set object 16 rect from 0.100202, 0 to 116.162217, 10 fc rgb "#00FF00"
set object 17 rect from 0.102398, 0 to 116.943100, 10 fc rgb "#0000FF"
set object 18 rect from 0.102855, 0 to 117.820661, 10 fc rgb "#FFFF00"
set object 19 rect from 0.103649, 0 to 119.059084, 10 fc rgb "#FF00FF"
set object 20 rect from 0.104723, 0 to 119.564453, 10 fc rgb "#808080"
set object 21 rect from 0.105181, 0 to 120.184750, 10 fc rgb "#800080"
set object 22 rect from 0.105957, 0 to 120.790325, 10 fc rgb "#008080"
set object 23 rect from 0.106240, 0 to 153.244683, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.132057, 10 to 153.925497, 20 fc rgb "#FF0000"
set object 25 rect from 0.068918, 10 to 80.659631, 20 fc rgb "#00FF00"
set object 26 rect from 0.071296, 10 to 81.720507, 20 fc rgb "#0000FF"
set object 27 rect from 0.071910, 10 to 82.610551, 20 fc rgb "#FFFF00"
set object 28 rect from 0.072727, 10 to 83.992396, 20 fc rgb "#FF00FF"
set object 29 rect from 0.073934, 10 to 84.550141, 20 fc rgb "#808080"
set object 30 rect from 0.074441, 10 to 85.280887, 20 fc rgb "#800080"
set object 31 rect from 0.075305, 10 to 85.954713, 20 fc rgb "#008080"
set object 32 rect from 0.075636, 10 to 149.596648, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.130415, 20 to 151.878850, 30 fc rgb "#FF0000"
set object 34 rect from 0.053334, 20 to 63.039820, 30 fc rgb "#00FF00"
set object 35 rect from 0.055749, 20 to 63.762629, 30 fc rgb "#0000FF"
set object 36 rect from 0.056129, 20 to 64.751726, 30 fc rgb "#FFFF00"
set object 37 rect from 0.057012, 20 to 66.123326, 30 fc rgb "#FF00FF"
set object 38 rect from 0.058231, 20 to 66.645724, 30 fc rgb "#808080"
set object 39 rect from 0.058681, 20 to 67.315004, 30 fc rgb "#800080"
set object 40 rect from 0.059536, 20 to 68.068547, 30 fc rgb "#008080"
set object 41 rect from 0.059932, 20 to 147.721985, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.128606, 30 to 153.896867, 40 fc rgb "#FF0000"
set object 43 rect from 0.030958, 30 to 38.508614, 40 fc rgb "#00FF00"
set object 44 rect from 0.034288, 30 to 40.202337, 40 fc rgb "#0000FF"
set object 45 rect from 0.035449, 30 to 43.459930, 40 fc rgb "#FFFF00"
set object 46 rect from 0.038351, 30 to 45.413157, 40 fc rgb "#FF00FF"
set object 47 rect from 0.040026, 30 to 46.215615, 40 fc rgb "#808080"
set object 48 rect from 0.040747, 30 to 47.047652, 40 fc rgb "#800080"
set object 49 rect from 0.041886, 30 to 48.242791, 40 fc rgb "#008080"
set object 50 rect from 0.042514, 30 to 145.203076, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.136754, 40 to 158.769756, 50 fc rgb "#FF0000"
set object 52 rect from 0.115714, 40 to 133.783249, 50 fc rgb "#00FF00"
set object 53 rect from 0.117879, 40 to 134.480956, 50 fc rgb "#0000FF"
set object 54 rect from 0.118261, 40 to 135.409739, 50 fc rgb "#FFFF00"
set object 55 rect from 0.119091, 40 to 136.651554, 50 fc rgb "#FF00FF"
set object 56 rect from 0.120171, 40 to 137.103462, 50 fc rgb "#808080"
set object 57 rect from 0.120567, 40 to 137.691941, 50 fc rgb "#800080"
set object 58 rect from 0.121331, 40 to 138.373704, 50 fc rgb "#008080"
set object 59 rect from 0.121688, 40 to 155.115954, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.133642, 50 to 155.455106, 60 fc rgb "#FF0000"
set object 61 rect from 0.085688, 50 to 99.582792, 60 fc rgb "#00FF00"
set object 62 rect from 0.087857, 50 to 100.404585, 60 fc rgb "#0000FF"
set object 63 rect from 0.088325, 50 to 101.378892, 60 fc rgb "#FFFF00"
set object 64 rect from 0.089198, 50 to 102.649133, 60 fc rgb "#FF00FF"
set object 65 rect from 0.090310, 50 to 103.147718, 60 fc rgb "#808080"
set object 66 rect from 0.090747, 50 to 103.780566, 60 fc rgb "#800080"
set object 67 rect from 0.091543, 50 to 104.505614, 60 fc rgb "#008080"
set object 68 rect from 0.091929, 50 to 151.520295, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
