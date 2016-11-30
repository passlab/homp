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

set object 15 rect from 0.112634, 0 to 156.470360, 10 fc rgb "#FF0000"
set object 16 rect from 0.072764, 0 to 99.904398, 10 fc rgb "#00FF00"
set object 17 rect from 0.074337, 0 to 100.708764, 10 fc rgb "#0000FF"
set object 18 rect from 0.074718, 0 to 101.460500, 10 fc rgb "#FFFF00"
set object 19 rect from 0.075275, 0 to 102.559066, 10 fc rgb "#FF00FF"
set object 20 rect from 0.076090, 0 to 104.556497, 10 fc rgb "#808080"
set object 21 rect from 0.077572, 0 to 105.182702, 10 fc rgb "#800080"
set object 22 rect from 0.078246, 0 to 105.777877, 10 fc rgb "#008080"
set object 23 rect from 0.078476, 0 to 151.380995, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.111365, 10 to 155.332619, 20 fc rgb "#FF0000"
set object 25 rect from 0.059790, 10 to 82.576851, 20 fc rgb "#00FF00"
set object 26 rect from 0.061480, 10 to 83.495928, 20 fc rgb "#0000FF"
set object 27 rect from 0.061979, 10 to 84.354271, 20 fc rgb "#FFFF00"
set object 28 rect from 0.062625, 10 to 85.674174, 20 fc rgb "#FF00FF"
set object 29 rect from 0.063613, 10 to 87.834904, 20 fc rgb "#808080"
set object 30 rect from 0.065203, 10 to 88.535338, 20 fc rgb "#800080"
set object 31 rect from 0.065924, 10 to 89.214193, 20 fc rgb "#008080"
set object 32 rect from 0.066206, 10 to 149.494233, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.109858, 20 to 154.856198, 30 fc rgb "#FF0000"
set object 34 rect from 0.045851, 20 to 63.901024, 30 fc rgb "#00FF00"
set object 35 rect from 0.047640, 20 to 65.011756, 30 fc rgb "#0000FF"
set object 36 rect from 0.048270, 20 to 67.349259, 30 fc rgb "#FFFF00"
set object 37 rect from 0.050027, 20 to 68.922898, 30 fc rgb "#FF00FF"
set object 38 rect from 0.051174, 20 to 70.623395, 30 fc rgb "#808080"
set object 39 rect from 0.052439, 20 to 71.330586, 30 fc rgb "#800080"
set object 40 rect from 0.053184, 20 to 72.071524, 30 fc rgb "#008080"
set object 41 rect from 0.053525, 20 to 147.626395, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.108441, 30 to 156.406911, 40 fc rgb "#FF0000"
set object 43 rect from 0.018036, 30 to 27.379399, 40 fc rgb "#00FF00"
set object 44 rect from 0.020694, 30 to 28.769488, 40 fc rgb "#0000FF"
set object 45 rect from 0.021425, 30 to 32.468764, 40 fc rgb "#FFFF00"
set object 46 rect from 0.024222, 30 to 35.674061, 40 fc rgb "#FF00FF"
set object 47 rect from 0.026541, 30 to 37.637747, 40 fc rgb "#808080"
set object 48 rect from 0.028016, 30 to 38.523079, 40 fc rgb "#800080"
set object 49 rect from 0.029053, 30 to 40.475965, 40 fc rgb "#008080"
set object 50 rect from 0.030104, 30 to 145.230833, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.115040, 40 to 160.920005, 50 fc rgb "#FF0000"
set object 52 rect from 0.097010, 40 to 132.383961, 50 fc rgb "#00FF00"
set object 53 rect from 0.098385, 40 to 133.008840, 50 fc rgb "#0000FF"
set object 54 rect from 0.098652, 40 to 134.975200, 50 fc rgb "#FFFF00"
set object 55 rect from 0.100108, 40 to 136.507009, 50 fc rgb "#FF00FF"
set object 56 rect from 0.101240, 40 to 138.133278, 50 fc rgb "#808080"
set object 57 rect from 0.102450, 40 to 138.837774, 50 fc rgb "#800080"
set object 58 rect from 0.103182, 40 to 139.485579, 50 fc rgb "#008080"
set object 59 rect from 0.103449, 40 to 154.740159, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.113824, 50 to 159.722877, 60 fc rgb "#FF0000"
set object 61 rect from 0.083981, 50 to 114.974081, 60 fc rgb "#00FF00"
set object 62 rect from 0.085485, 50 to 115.737944, 60 fc rgb "#0000FF"
set object 63 rect from 0.085854, 50 to 117.908126, 60 fc rgb "#FFFF00"
set object 64 rect from 0.087481, 50 to 119.485807, 60 fc rgb "#FF00FF"
set object 65 rect from 0.088645, 50 to 121.172810, 60 fc rgb "#808080"
set object 66 rect from 0.089891, 50 to 121.931284, 60 fc rgb "#800080"
set object 67 rect from 0.090653, 50 to 122.651969, 60 fc rgb "#008080"
set object 68 rect from 0.090996, 50 to 153.084187, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
