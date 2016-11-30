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

set object 15 rect from 0.109921, 0 to 156.664706, 10 fc rgb "#FF0000"
set object 16 rect from 0.070272, 0 to 99.517581, 10 fc rgb "#00FF00"
set object 17 rect from 0.072071, 0 to 100.332958, 10 fc rgb "#0000FF"
set object 18 rect from 0.072453, 0 to 101.284259, 10 fc rgb "#FFFF00"
set object 19 rect from 0.073168, 0 to 102.609938, 10 fc rgb "#FF00FF"
set object 20 rect from 0.074106, 0 to 103.977192, 10 fc rgb "#808080"
set object 21 rect from 0.075094, 0 to 104.662224, 10 fc rgb "#800080"
set object 22 rect from 0.075810, 0 to 105.348662, 10 fc rgb "#008080"
set object 23 rect from 0.076089, 0 to 151.622678, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.111303, 10 to 158.126268, 20 fc rgb "#FF0000"
set object 25 rect from 0.082524, 10 to 116.092749, 20 fc rgb "#00FF00"
set object 26 rect from 0.084021, 10 to 116.838821, 20 fc rgb "#0000FF"
set object 27 rect from 0.084357, 10 to 117.690234, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084971, 10 to 118.889743, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085861, 10 to 120.096154, 20 fc rgb "#808080"
set object 30 rect from 0.086725, 10 to 120.771474, 20 fc rgb "#800080"
set object 31 rect from 0.087425, 10 to 121.398277, 20 fc rgb "#008080"
set object 32 rect from 0.087647, 10 to 153.704099, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.105588, 20 to 153.570945, 30 fc rgb "#FF0000"
set object 34 rect from 0.025378, 20 to 38.496032, 30 fc rgb "#00FF00"
set object 35 rect from 0.028205, 20 to 39.977059, 30 fc rgb "#0000FF"
set object 36 rect from 0.028950, 20 to 42.320575, 30 fc rgb "#FFFF00"
set object 37 rect from 0.030665, 20 to 44.644667, 30 fc rgb "#FF00FF"
set object 38 rect from 0.032332, 20 to 45.931500, 30 fc rgb "#808080"
set object 39 rect from 0.033259, 20 to 46.856434, 30 fc rgb "#800080"
set object 40 rect from 0.034308, 20 to 48.428958, 30 fc rgb "#008080"
set object 41 rect from 0.035040, 20 to 145.257738, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.112562, 30 to 160.053789, 40 fc rgb "#FF0000"
set object 43 rect from 0.094470, 30 to 132.583362, 40 fc rgb "#00FF00"
set object 44 rect from 0.095934, 30 to 133.351584, 40 fc rgb "#0000FF"
set object 45 rect from 0.096265, 30 to 134.315324, 40 fc rgb "#FFFF00"
set object 46 rect from 0.096986, 30 to 135.788003, 40 fc rgb "#FF00FF"
set object 47 rect from 0.098032, 30 to 136.825263, 40 fc rgb "#808080"
set object 48 rect from 0.098780, 30 to 137.474217, 40 fc rgb "#800080"
set object 49 rect from 0.099457, 30 to 138.236902, 40 fc rgb "#008080"
set object 50 rect from 0.099790, 30 to 155.481852, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.107072, 40 to 153.448948, 50 fc rgb "#FF0000"
set object 52 rect from 0.043584, 40 to 62.691108, 50 fc rgb "#00FF00"
set object 53 rect from 0.045530, 40 to 63.702002, 50 fc rgb "#0000FF"
set object 54 rect from 0.046038, 40 to 64.783647, 50 fc rgb "#FFFF00"
set object 55 rect from 0.046824, 40 to 66.360304, 50 fc rgb "#FF00FF"
set object 56 rect from 0.047967, 40 to 67.862118, 50 fc rgb "#808080"
set object 57 rect from 0.049065, 40 to 68.588725, 50 fc rgb "#800080"
set object 58 rect from 0.049812, 40 to 69.417987, 50 fc rgb "#008080"
set object 59 rect from 0.050178, 40 to 147.730235, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.108452, 50 to 154.075792, 60 fc rgb "#FF0000"
set object 61 rect from 0.057153, 50 to 81.167430, 60 fc rgb "#00FF00"
set object 62 rect from 0.058843, 50 to 81.901021, 60 fc rgb "#0000FF"
set object 63 rect from 0.059179, 50 to 82.778801, 60 fc rgb "#FFFF00"
set object 64 rect from 0.059799, 50 to 84.044846, 60 fc rgb "#FF00FF"
set object 65 rect from 0.060706, 50 to 85.083470, 60 fc rgb "#808080"
set object 66 rect from 0.061460, 50 to 85.728292, 60 fc rgb "#800080"
set object 67 rect from 0.062177, 50 to 86.502052, 60 fc rgb "#008080"
set object 68 rect from 0.062480, 50 to 149.592542, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
