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

set object 15 rect from 0.134082, 0 to 160.140547, 10 fc rgb "#FF0000"
set object 16 rect from 0.112293, 0 to 132.913410, 10 fc rgb "#00FF00"
set object 17 rect from 0.114569, 0 to 133.682616, 10 fc rgb "#0000FF"
set object 18 rect from 0.114987, 0 to 134.547247, 10 fc rgb "#FFFF00"
set object 19 rect from 0.115741, 0 to 135.767970, 10 fc rgb "#FF00FF"
set object 20 rect from 0.116792, 0 to 137.205143, 10 fc rgb "#808080"
set object 21 rect from 0.118032, 0 to 137.839361, 10 fc rgb "#800080"
set object 22 rect from 0.118832, 0 to 138.501507, 10 fc rgb "#008080"
set object 23 rect from 0.119157, 0 to 155.374026, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.130959, 10 to 156.882183, 20 fc rgb "#FF0000"
set object 25 rect from 0.082684, 10 to 98.425931, 20 fc rgb "#00FF00"
set object 26 rect from 0.084943, 10 to 99.361547, 20 fc rgb "#0000FF"
set object 27 rect from 0.085498, 10 to 100.272726, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086309, 10 to 101.628440, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087487, 10 to 103.137762, 20 fc rgb "#808080"
set object 30 rect from 0.088779, 10 to 103.853438, 20 fc rgb "#800080"
set object 31 rect from 0.089628, 10 to 104.537695, 20 fc rgb "#008080"
set object 32 rect from 0.089967, 10 to 151.618759, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129210, 20 to 154.905055, 30 fc rgb "#FF0000"
set object 34 rect from 0.066718, 20 to 79.724073, 30 fc rgb "#00FF00"
set object 35 rect from 0.068872, 20 to 80.604996, 30 fc rgb "#0000FF"
set object 36 rect from 0.069378, 20 to 81.574360, 30 fc rgb "#FFFF00"
set object 37 rect from 0.070212, 20 to 82.996403, 30 fc rgb "#FF00FF"
set object 38 rect from 0.071433, 20 to 84.364918, 30 fc rgb "#808080"
set object 39 rect from 0.072612, 20 to 85.081759, 30 fc rgb "#800080"
set object 40 rect from 0.073511, 20 to 85.831182, 30 fc rgb "#008080"
set object 41 rect from 0.073877, 20 to 149.691667, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.127611, 30 to 153.360820, 40 fc rgb "#FF0000"
set object 43 rect from 0.050539, 30 to 61.208407, 40 fc rgb "#00FF00"
set object 44 rect from 0.052948, 30 to 62.103294, 40 fc rgb "#0000FF"
set object 45 rect from 0.053484, 30 to 63.254196, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054469, 30 to 64.787956, 40 fc rgb "#FF00FF"
set object 47 rect from 0.055801, 30 to 66.183234, 40 fc rgb "#808080"
set object 48 rect from 0.057016, 30 to 66.925677, 40 fc rgb "#800080"
set object 49 rect from 0.057916, 30 to 67.708848, 40 fc rgb "#008080"
set object 50 rect from 0.058324, 30 to 147.808797, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.125817, 40 to 153.874014, 50 fc rgb "#FF0000"
set object 52 rect from 0.029139, 40 to 37.233815, 50 fc rgb "#00FF00"
set object 53 rect from 0.032425, 40 to 38.659350, 50 fc rgb "#0000FF"
set object 54 rect from 0.033362, 40 to 41.213676, 50 fc rgb "#FFFF00"
set object 55 rect from 0.035574, 40 to 43.245500, 50 fc rgb "#FF00FF"
set object 56 rect from 0.037295, 40 to 44.813007, 50 fc rgb "#808080"
set object 57 rect from 0.038648, 40 to 45.709058, 50 fc rgb "#800080"
set object 58 rect from 0.039883, 40 to 47.189287, 50 fc rgb "#008080"
set object 59 rect from 0.040692, 40 to 145.227706, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.132542, 50 to 158.533475, 60 fc rgb "#FF0000"
set object 61 rect from 0.097440, 50 to 115.292631, 60 fc rgb "#00FF00"
set object 62 rect from 0.099432, 50 to 115.989688, 60 fc rgb "#0000FF"
set object 63 rect from 0.099786, 50 to 116.974180, 60 fc rgb "#FFFF00"
set object 64 rect from 0.100633, 50 to 118.341531, 60 fc rgb "#FF00FF"
set object 65 rect from 0.101841, 50 to 119.704225, 60 fc rgb "#808080"
set object 66 rect from 0.102994, 50 to 120.457141, 60 fc rgb "#800080"
set object 67 rect from 0.103898, 50 to 121.253114, 60 fc rgb "#008080"
set object 68 rect from 0.104313, 50 to 153.583088, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
