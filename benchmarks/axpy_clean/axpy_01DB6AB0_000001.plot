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

set object 15 rect from 0.095843, 0 to 162.478345, 10 fc rgb "#FF0000"
set object 16 rect from 0.079102, 0 to 133.212256, 10 fc rgb "#00FF00"
set object 17 rect from 0.080607, 0 to 134.059934, 10 fc rgb "#0000FF"
set object 18 rect from 0.080913, 0 to 135.027055, 10 fc rgb "#FFFF00"
set object 19 rect from 0.081504, 0 to 136.435468, 10 fc rgb "#FF00FF"
set object 20 rect from 0.082345, 0 to 136.989529, 10 fc rgb "#808080"
set object 21 rect from 0.082690, 0 to 137.732742, 10 fc rgb "#800080"
set object 22 rect from 0.083365, 0 to 138.519017, 10 fc rgb "#008080"
set object 23 rect from 0.083604, 0 to 158.153686, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.094543, 10 to 160.678526, 20 fc rgb "#FF0000"
set object 25 rect from 0.067183, 10 to 113.718636, 20 fc rgb "#00FF00"
set object 26 rect from 0.068860, 10 to 114.629348, 20 fc rgb "#0000FF"
set object 27 rect from 0.069199, 10 to 115.599782, 20 fc rgb "#FFFF00"
set object 28 rect from 0.069813, 10 to 117.175743, 20 fc rgb "#FF00FF"
set object 29 rect from 0.070759, 10 to 117.821067, 20 fc rgb "#808080"
set object 30 rect from 0.071151, 10 to 118.700287, 20 fc rgb "#800080"
set object 31 rect from 0.071884, 10 to 119.538027, 20 fc rgb "#008080"
set object 32 rect from 0.072164, 10 to 155.973880, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.093218, 20 to 158.579898, 30 fc rgb "#FF0000"
set object 34 rect from 0.056157, 20 to 95.022908, 30 fc rgb "#00FF00"
set object 35 rect from 0.057586, 20 to 95.872267, 30 fc rgb "#0000FF"
set object 36 rect from 0.057893, 20 to 96.995318, 30 fc rgb "#FFFF00"
set object 37 rect from 0.058575, 20 to 98.508245, 30 fc rgb "#FF00FF"
set object 38 rect from 0.059482, 20 to 99.181749, 30 fc rgb "#808080"
set object 39 rect from 0.059890, 20 to 99.979692, 30 fc rgb "#800080"
set object 40 rect from 0.060594, 20 to 100.835675, 30 fc rgb "#008080"
set object 41 rect from 0.060891, 20 to 153.893643, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.091886, 30 to 156.189581, 40 fc rgb "#FF0000"
set object 43 rect from 0.044636, 30 to 76.162994, 40 fc rgb "#00FF00"
set object 44 rect from 0.046221, 30 to 76.972505, 40 fc rgb "#0000FF"
set object 45 rect from 0.046501, 30 to 78.173571, 40 fc rgb "#FFFF00"
set object 46 rect from 0.047252, 30 to 79.686498, 40 fc rgb "#FF00FF"
set object 47 rect from 0.048166, 30 to 80.268738, 40 fc rgb "#808080"
set object 48 rect from 0.048508, 30 to 81.066681, 40 fc rgb "#800080"
set object 49 rect from 0.049225, 30 to 82.002310, 40 fc rgb "#008080"
set object 50 rect from 0.049541, 30 to 151.644178, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.090426, 40 to 153.989851, 50 fc rgb "#FF0000"
set object 52 rect from 0.032956, 40 to 57.253296, 50 fc rgb "#00FF00"
set object 53 rect from 0.034839, 40 to 58.208799, 50 fc rgb "#0000FF"
set object 54 rect from 0.035198, 40 to 59.321914, 50 fc rgb "#FFFF00"
set object 55 rect from 0.035881, 40 to 60.979152, 50 fc rgb "#FF00FF"
set object 56 rect from 0.036882, 40 to 61.569748, 50 fc rgb "#808080"
set object 57 rect from 0.037215, 40 to 62.357703, 50 fc rgb "#800080"
set object 58 rect from 0.037959, 40 to 63.318200, 50 fc rgb "#008080"
set object 59 rect from 0.038300, 40 to 149.094471, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.088878, 50 to 154.804306, 60 fc rgb "#FF0000"
set object 61 rect from 0.016972, 50 to 31.822520, 60 fc rgb "#00FF00"
set object 62 rect from 0.019591, 50 to 33.406737, 60 fc rgb "#0000FF"
set object 63 rect from 0.020260, 50 to 36.017847, 60 fc rgb "#FFFF00"
set object 64 rect from 0.021844, 50 to 38.582485, 60 fc rgb "#FF00FF"
set object 65 rect from 0.023382, 50 to 39.433475, 60 fc rgb "#808080"
set object 66 rect from 0.023899, 50 to 40.516729, 60 fc rgb "#800080"
set object 67 rect from 0.024922, 50 to 42.160717, 60 fc rgb "#008080"
set object 68 rect from 0.025533, 50 to 145.619071, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
