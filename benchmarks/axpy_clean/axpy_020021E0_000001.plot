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

set object 15 rect from 0.187760, 0 to 150.518836, 10 fc rgb "#FF0000"
set object 16 rect from 0.169187, 0 to 130.167623, 10 fc rgb "#00FF00"
set object 17 rect from 0.171532, 0 to 130.676384, 10 fc rgb "#0000FF"
set object 18 rect from 0.171948, 0 to 131.284768, 10 fc rgb "#FFFF00"
set object 19 rect from 0.172769, 0 to 132.115973, 10 fc rgb "#FF00FF"
set object 20 rect from 0.173865, 0 to 138.034031, 10 fc rgb "#808080"
set object 21 rect from 0.181644, 0 to 138.483476, 10 fc rgb "#800080"
set object 22 rect from 0.182488, 0 to 138.907063, 10 fc rgb "#008080"
set object 23 rect from 0.182794, 0 to 142.064577, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.196387, 10 to 157.421717, 20 fc rgb "#FF0000"
set object 25 rect from 0.119945, 10 to 92.826516, 20 fc rgb "#00FF00"
set object 26 rect from 0.122435, 10 to 93.380905, 20 fc rgb "#0000FF"
set object 27 rect from 0.122905, 10 to 94.084350, 20 fc rgb "#FFFF00"
set object 28 rect from 0.123869, 10 to 95.005292, 20 fc rgb "#FF00FF"
set object 29 rect from 0.125096, 10 to 101.051110, 20 fc rgb "#808080"
set object 30 rect from 0.133028, 10 to 101.540860, 20 fc rgb "#800080"
set object 31 rect from 0.133940, 10 to 102.026807, 20 fc rgb "#008080"
set object 32 rect from 0.134298, 10 to 148.838176, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.194647, 20 to 159.653727, 30 fc rgb "#FF0000"
set object 34 rect from 0.091471, 20 to 71.110238, 30 fc rgb "#00FF00"
set object 35 rect from 0.093865, 20 to 71.649419, 30 fc rgb "#0000FF"
set object 36 rect from 0.094329, 20 to 74.114135, 30 fc rgb "#FFFF00"
set object 37 rect from 0.097575, 20 to 76.755284, 30 fc rgb "#FF00FF"
set object 38 rect from 0.101050, 20 to 82.766121, 30 fc rgb "#808080"
set object 39 rect from 0.108979, 20 to 83.298457, 30 fc rgb "#800080"
set object 40 rect from 0.109940, 20 to 83.825469, 30 fc rgb "#008080"
set object 41 rect from 0.110355, 20 to 147.645743, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.191081, 30 to 159.290218, 40 fc rgb "#FF0000"
set object 43 rect from 0.030334, 30 to 25.646440, 40 fc rgb "#00FF00"
set object 44 rect from 0.034198, 30 to 26.685256, 40 fc rgb "#0000FF"
set object 45 rect from 0.035231, 30 to 30.232137, 40 fc rgb "#FFFF00"
set object 46 rect from 0.039913, 30 to 33.499161, 40 fc rgb "#FF00FF"
set object 47 rect from 0.044181, 30 to 39.722171, 40 fc rgb "#808080"
set object 48 rect from 0.052646, 30 to 40.538927, 40 fc rgb "#800080"
set object 49 rect from 0.053908, 30 to 41.519186, 40 fc rgb "#008080"
set object 50 rect from 0.054733, 30 to 144.830445, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.193099, 40 to 158.274216, 50 fc rgb "#FF0000"
set object 52 rect from 0.064286, 40 to 50.672328, 50 fc rgb "#00FF00"
set object 53 rect from 0.066989, 40 to 51.194017, 50 fc rgb "#0000FF"
set object 54 rect from 0.067434, 40 to 53.580405, 50 fc rgb "#FFFF00"
set object 55 rect from 0.070576, 40 to 56.189614, 50 fc rgb "#FF00FF"
set object 56 rect from 0.074018, 40 to 62.148737, 50 fc rgb "#808080"
set object 57 rect from 0.081868, 40 to 62.671187, 50 fc rgb "#800080"
set object 58 rect from 0.082810, 40 to 63.214930, 50 fc rgb "#008080"
set object 59 rect from 0.083279, 40 to 146.197028, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.198140, 50 to 162.116924, 60 fc rgb "#FF0000"
set object 61 rect from 0.141781, 50 to 109.153268, 60 fc rgb "#00FF00"
set object 62 rect from 0.143903, 50 to 109.761652, 60 fc rgb "#0000FF"
set object 63 rect from 0.144459, 50 to 112.178459, 60 fc rgb "#FFFF00"
set object 64 rect from 0.147622, 50 to 114.789189, 60 fc rgb "#FF00FF"
set object 65 rect from 0.151074, 50 to 120.641846, 60 fc rgb "#808080"
set object 66 rect from 0.158780, 50 to 121.171900, 60 fc rgb "#800080"
set object 67 rect from 0.159741, 50 to 121.699673, 60 fc rgb "#008080"
set object 68 rect from 0.160148, 50 to 150.262556, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
