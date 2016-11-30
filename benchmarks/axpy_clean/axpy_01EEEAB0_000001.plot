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

set object 15 rect from 0.120242, 0 to 157.295255, 10 fc rgb "#FF0000"
set object 16 rect from 0.103707, 0 to 134.907172, 10 fc rgb "#00FF00"
set object 17 rect from 0.105214, 0 to 135.579716, 10 fc rgb "#0000FF"
set object 18 rect from 0.105533, 0 to 136.299837, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106095, 0 to 137.395451, 10 fc rgb "#FF00FF"
set object 20 rect from 0.106950, 0 to 137.801808, 10 fc rgb "#808080"
set object 21 rect from 0.107262, 0 to 138.366331, 10 fc rgb "#800080"
set object 22 rect from 0.107916, 0 to 138.965576, 10 fc rgb "#008080"
set object 23 rect from 0.108182, 0 to 154.013556, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.118934, 10 to 156.105769, 20 fc rgb "#FF0000"
set object 25 rect from 0.091812, 10 to 119.902912, 20 fc rgb "#00FF00"
set object 26 rect from 0.093548, 10 to 120.637181, 20 fc rgb "#0000FF"
set object 27 rect from 0.093913, 10 to 121.389450, 20 fc rgb "#FFFF00"
set object 28 rect from 0.094538, 10 to 122.720392, 20 fc rgb "#FF00FF"
set object 29 rect from 0.095541, 10 to 123.242479, 20 fc rgb "#808080"
set object 30 rect from 0.095950, 10 to 123.924024, 20 fc rgb "#800080"
set object 31 rect from 0.096688, 10 to 124.561847, 20 fc rgb "#008080"
set object 32 rect from 0.096982, 10 to 152.317413, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.117635, 20 to 154.243741, 30 fc rgb "#FF0000"
set object 34 rect from 0.081357, 20 to 106.160153, 30 fc rgb "#00FF00"
set object 35 rect from 0.082865, 20 to 106.801834, 30 fc rgb "#0000FF"
set object 36 rect from 0.083152, 20 to 107.700699, 30 fc rgb "#FFFF00"
set object 37 rect from 0.083879, 20 to 108.933910, 30 fc rgb "#FF00FF"
set object 38 rect from 0.084818, 20 to 109.390416, 30 fc rgb "#808080"
set object 39 rect from 0.085168, 20 to 109.960084, 30 fc rgb "#800080"
set object 40 rect from 0.085828, 20 to 110.588903, 30 fc rgb "#008080"
set object 41 rect from 0.086117, 20 to 150.765291, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.116355, 30 to 152.707049, 40 fc rgb "#FF0000"
set object 43 rect from 0.069921, 30 to 91.636832, 40 fc rgb "#00FF00"
set object 44 rect from 0.071580, 30 to 92.286228, 40 fc rgb "#0000FF"
set object 45 rect from 0.071867, 30 to 93.361268, 40 fc rgb "#FFFF00"
set object 46 rect from 0.072702, 30 to 94.507034, 40 fc rgb "#FF00FF"
set object 47 rect from 0.073602, 30 to 94.933963, 40 fc rgb "#808080"
set object 48 rect from 0.073927, 30 to 95.530635, 40 fc rgb "#800080"
set object 49 rect from 0.074606, 30 to 96.191605, 40 fc rgb "#008080"
set object 50 rect from 0.074920, 30 to 149.036999, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.113572, 40 to 184.018218, 50 fc rgb "#FF0000"
set object 52 rect from 0.014197, 40 to 21.181880, 50 fc rgb "#00FF00"
set object 53 rect from 0.016858, 40 to 22.444665, 50 fc rgb "#0000FF"
set object 54 rect from 0.017568, 40 to 56.202963, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043883, 40 to 58.581938, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045685, 40 to 59.340638, 50 fc rgb "#808080"
set object 57 rect from 0.046268, 40 to 60.206071, 50 fc rgb "#800080"
set object 58 rect from 0.047320, 40 to 61.443138, 50 fc rgb "#008080"
set object 59 rect from 0.047896, 40 to 145.122621, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.115057, 50 to 151.289953, 60 fc rgb "#FF0000"
set object 61 rect from 0.058276, 50 to 77.063360, 60 fc rgb "#00FF00"
set object 62 rect from 0.060264, 50 to 77.827203, 60 fc rgb "#0000FF"
set object 63 rect from 0.060621, 50 to 78.836661, 60 fc rgb "#FFFF00"
set object 64 rect from 0.061423, 50 to 80.176601, 60 fc rgb "#FF00FF"
set object 65 rect from 0.062479, 50 to 80.680688, 60 fc rgb "#808080"
set object 66 rect from 0.062852, 50 to 81.288934, 60 fc rgb "#800080"
set object 67 rect from 0.063550, 50 to 82.006485, 60 fc rgb "#008080"
set object 68 rect from 0.063900, 50 to 147.245695, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
