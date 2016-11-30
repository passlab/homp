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

set object 15 rect from 0.222288, 0 to 157.303947, 10 fc rgb "#FF0000"
set object 16 rect from 0.044992, 0 to 32.350209, 10 fc rgb "#00FF00"
set object 17 rect from 0.049912, 0 to 35.005496, 10 fc rgb "#0000FF"
set object 18 rect from 0.053561, 0 to 36.444185, 10 fc rgb "#FFFF00"
set object 19 rect from 0.055791, 0 to 37.569717, 10 fc rgb "#FF00FF"
set object 20 rect from 0.057498, 0 to 44.239053, 10 fc rgb "#808080"
set object 21 rect from 0.067765, 0 to 44.736305, 10 fc rgb "#800080"
set object 22 rect from 0.068849, 0 to 45.266970, 10 fc rgb "#008080"
set object 23 rect from 0.069231, 0 to 144.917852, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.224224, 10 to 155.545550, 20 fc rgb "#FF0000"
set object 25 rect from 0.079043, 10 to 53.367134, 20 fc rgb "#00FF00"
set object 26 rect from 0.081852, 10 to 54.550974, 20 fc rgb "#0000FF"
set object 27 rect from 0.083385, 10 to 55.158945, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084311, 10 to 55.865841, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085403, 10 to 62.157161, 20 fc rgb "#808080"
set object 30 rect from 0.095006, 10 to 62.520764, 20 fc rgb "#800080"
set object 31 rect from 0.095855, 10 to 62.892229, 20 fc rgb "#008080"
set object 32 rect from 0.096128, 10 to 146.473157, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.231185, 20 to 162.229299, 30 fc rgb "#FF0000"
set object 34 rect from 0.194329, 20 to 128.479709, 30 fc rgb "#00FF00"
set object 35 rect from 0.196461, 20 to 128.857725, 30 fc rgb "#0000FF"
set object 36 rect from 0.196801, 20 to 130.836906, 30 fc rgb "#FFFF00"
set object 37 rect from 0.199827, 20 to 133.058490, 30 fc rgb "#FF00FF"
set object 38 rect from 0.203216, 20 to 139.274468, 30 fc rgb "#808080"
set object 39 rect from 0.212786, 20 to 139.765168, 30 fc rgb "#800080"
set object 40 rect from 0.213708, 20 to 140.158253, 30 fc rgb "#008080"
set object 41 rect from 0.214059, 20 to 151.118107, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.226103, 30 to 159.760073, 40 fc rgb "#FF0000"
set object 43 rect from 0.104039, 30 to 69.524222, 40 fc rgb "#00FF00"
set object 44 rect from 0.106530, 30 to 69.999200, 40 fc rgb "#0000FF"
set object 45 rect from 0.106980, 30 to 72.084513, 40 fc rgb "#FFFF00"
set object 46 rect from 0.110184, 30 to 74.805970, 40 fc rgb "#FF00FF"
set object 47 rect from 0.114314, 30 to 81.165422, 40 fc rgb "#808080"
set object 48 rect from 0.124026, 30 to 81.699363, 40 fc rgb "#800080"
set object 49 rect from 0.125097, 30 to 82.670282, 40 fc rgb "#008080"
set object 50 rect from 0.126310, 30 to 147.664859, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.227963, 40 to 160.169536, 50 fc rgb "#FF0000"
set object 52 rect from 0.136663, 40 to 90.697725, 50 fc rgb "#00FF00"
set object 53 rect from 0.138795, 40 to 91.115705, 50 fc rgb "#0000FF"
set object 54 rect from 0.139192, 40 to 93.052302, 50 fc rgb "#FFFF00"
set object 55 rect from 0.142170, 40 to 95.283713, 50 fc rgb "#FF00FF"
set object 56 rect from 0.145567, 40 to 101.552756, 50 fc rgb "#808080"
set object 57 rect from 0.155206, 40 to 102.052630, 50 fc rgb "#800080"
set object 58 rect from 0.156164, 40 to 102.455541, 50 fc rgb "#008080"
set object 59 rect from 0.156504, 40 to 148.899799, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.229595, 50 to 161.287860, 60 fc rgb "#FF0000"
set object 61 rect from 0.165131, 50 to 109.275560, 60 fc rgb "#00FF00"
set object 62 rect from 0.167146, 50 to 109.728261, 60 fc rgb "#0000FF"
set object 63 rect from 0.167601, 50 to 111.667479, 60 fc rgb "#FFFF00"
set object 64 rect from 0.170567, 50 to 113.813721, 60 fc rgb "#FF00FF"
set object 65 rect from 0.173922, 50 to 120.217725, 60 fc rgb "#808080"
set object 66 rect from 0.183618, 50 to 120.659290, 60 fc rgb "#800080"
set object 67 rect from 0.184555, 50 to 121.058271, 60 fc rgb "#008080"
set object 68 rect from 0.184920, 50 to 150.030572, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
