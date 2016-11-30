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

set object 15 rect from 0.377466, 0 to 151.338600, 10 fc rgb "#FF0000"
set object 16 rect from 0.112578, 0 to 44.476751, 10 fc rgb "#00FF00"
set object 17 rect from 0.114791, 0 to 44.753953, 10 fc rgb "#0000FF"
set object 18 rect from 0.115238, 0 to 45.006273, 10 fc rgb "#FFFF00"
set object 19 rect from 0.115905, 0 to 45.422652, 10 fc rgb "#FF00FF"
set object 20 rect from 0.116976, 0 to 49.160767, 10 fc rgb "#808080"
set object 21 rect from 0.126598, 0 to 49.390922, 10 fc rgb "#800080"
set object 22 rect from 0.127409, 0 to 49.586868, 10 fc rgb "#008080"
set object 23 rect from 0.127671, 0 to 146.492853, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.375175, 10 to 151.659356, 20 fc rgb "#FF0000"
set object 25 rect from 0.082628, 10 to 32.853417, 20 fc rgb "#00FF00"
set object 26 rect from 0.084857, 10 to 33.107290, 20 fc rgb "#0000FF"
set object 27 rect from 0.085280, 10 to 33.359216, 20 fc rgb "#FFFF00"
set object 28 rect from 0.085952, 10 to 33.896512, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087348, 10 to 38.700658, 20 fc rgb "#808080"
set object 30 rect from 0.099750, 10 to 38.995355, 20 fc rgb "#800080"
set object 31 rect from 0.100864, 10 to 39.307155, 20 fc rgb "#008080"
set object 32 rect from 0.101269, 10 to 145.591665, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.379293, 20 to 180.395587, 30 fc rgb "#FF0000"
set object 34 rect from 0.137640, 20 to 54.146474, 30 fc rgb "#00FF00"
set object 35 rect from 0.139622, 20 to 54.396459, 30 fc rgb "#0000FF"
set object 36 rect from 0.140045, 20 to 80.639849, 30 fc rgb "#FFFF00"
set object 37 rect from 0.207673, 20 to 83.459662, 30 fc rgb "#FF00FF"
set object 38 rect from 0.214817, 20 to 87.099026, 30 fc rgb "#808080"
set object 39 rect from 0.224185, 20 to 87.440376, 30 fc rgb "#800080"
set object 40 rect from 0.225297, 20 to 87.812049, 30 fc rgb "#008080"
set object 41 rect from 0.226002, 20 to 147.187217, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.382345, 30 to 164.513167, 40 fc rgb "#FF0000"
set object 43 rect from 0.320202, 30 to 125.034179, 40 fc rgb "#00FF00"
set object 44 rect from 0.321934, 30 to 125.274443, 40 fc rgb "#0000FF"
set object 45 rect from 0.322344, 30 to 134.974876, 40 fc rgb "#FFFF00"
set object 46 rect from 0.347423, 30 to 137.253512, 40 fc rgb "#FF00FF"
set object 47 rect from 0.353147, 30 to 140.941475, 40 fc rgb "#808080"
set object 48 rect from 0.362654, 30 to 141.221394, 40 fc rgb "#800080"
set object 49 rect from 0.363636, 30 to 141.532417, 40 fc rgb "#008080"
set object 50 rect from 0.364154, 30 to 148.461636, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.380982, 40 to 164.144218, 50 fc rgb "#FF0000"
set object 52 rect from 0.264040, 40 to 103.267975, 50 fc rgb "#00FF00"
set object 53 rect from 0.266023, 40 to 103.604656, 50 fc rgb "#0000FF"
set object 54 rect from 0.266595, 40 to 113.412780, 50 fc rgb "#FFFF00"
set object 55 rect from 0.291923, 40 to 115.699196, 50 fc rgb "#FF00FF"
set object 56 rect from 0.297703, 40 to 119.331562, 50 fc rgb "#808080"
set object 57 rect from 0.307070, 40 to 119.623924, 50 fc rgb "#800080"
set object 58 rect from 0.308072, 40 to 119.933777, 50 fc rgb "#008080"
set object 59 rect from 0.308600, 40 to 147.897517, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.372894, 50 to 153.551147, 60 fc rgb "#FF0000"
set object 61 rect from 0.042688, 50 to 17.816206, 60 fc rgb "#00FF00"
set object 62 rect from 0.046325, 50 to 18.349221, 60 fc rgb "#0000FF"
set object 63 rect from 0.047327, 50 to 20.011644, 60 fc rgb "#FFFF00"
set object 64 rect from 0.051630, 50 to 22.607525, 60 fc rgb "#FF00FF"
set object 65 rect from 0.058272, 50 to 26.351081, 60 fc rgb "#808080"
set object 66 rect from 0.067905, 50 to 26.748412, 60 fc rgb "#800080"
set object 67 rect from 0.069310, 50 to 27.203675, 60 fc rgb "#008080"
set object 68 rect from 0.070106, 50 to 144.531462, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
