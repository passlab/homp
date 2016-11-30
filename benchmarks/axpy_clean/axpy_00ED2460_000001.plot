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

set object 15 rect from 0.247139, 0 to 158.923391, 10 fc rgb "#FF0000"
set object 16 rect from 0.215837, 0 to 131.882189, 10 fc rgb "#00FF00"
set object 17 rect from 0.217390, 0 to 132.241847, 10 fc rgb "#0000FF"
set object 18 rect from 0.217772, 0 to 132.588140, 10 fc rgb "#FFFF00"
set object 19 rect from 0.218347, 0 to 133.094216, 10 fc rgb "#FF00FF"
set object 20 rect from 0.219183, 0 to 140.780693, 10 fc rgb "#808080"
set object 21 rect from 0.231839, 0 to 141.051045, 10 fc rgb "#800080"
set object 22 rect from 0.232524, 0 to 141.336587, 10 fc rgb "#008080"
set object 23 rect from 0.232746, 0 to 149.845662, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.245841, 10 to 158.484139, 20 fc rgb "#FF0000"
set object 25 rect from 0.185351, 10 to 113.439377, 20 fc rgb "#00FF00"
set object 26 rect from 0.187029, 10 to 113.792350, 20 fc rgb "#0000FF"
set object 27 rect from 0.187402, 10 to 114.212157, 20 fc rgb "#FFFF00"
set object 28 rect from 0.188136, 10 to 114.840950, 20 fc rgb "#FF00FF"
set object 29 rect from 0.189142, 10 to 122.679919, 20 fc rgb "#808080"
set object 30 rect from 0.202056, 10 to 122.984902, 20 fc rgb "#800080"
set object 31 rect from 0.202811, 10 to 123.316615, 20 fc rgb "#008080"
set object 32 rect from 0.203084, 10 to 149.024888, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.242816, 20 to 162.433093, 30 fc rgb "#FF0000"
set object 34 rect from 0.116729, 20 to 71.691678, 30 fc rgb "#00FF00"
set object 35 rect from 0.118306, 20 to 72.018529, 30 fc rgb "#0000FF"
set object 36 rect from 0.118642, 20 to 73.631523, 30 fc rgb "#FFFF00"
set object 37 rect from 0.121322, 20 to 79.079862, 30 fc rgb "#FF00FF"
set object 38 rect from 0.130283, 20 to 86.523327, 30 fc rgb "#808080"
set object 39 rect from 0.142530, 20 to 87.003887, 30 fc rgb "#800080"
set object 40 rect from 0.143536, 20 to 87.331951, 30 fc rgb "#008080"
set object 41 rect from 0.143860, 20 to 147.207154, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.241057, 30 to 161.230789, 40 fc rgb "#FF0000"
set object 43 rect from 0.080955, 30 to 50.069048, 40 fc rgb "#00FF00"
set object 44 rect from 0.082720, 30 to 50.544136, 40 fc rgb "#0000FF"
set object 45 rect from 0.083310, 30 to 52.195406, 40 fc rgb "#FFFF00"
set object 46 rect from 0.086031, 30 to 57.278618, 40 fc rgb "#FF00FF"
set object 47 rect from 0.094391, 30 to 64.778587, 40 fc rgb "#808080"
set object 48 rect from 0.106751, 30 to 65.259749, 40 fc rgb "#800080"
set object 49 rect from 0.107760, 30 to 65.627307, 40 fc rgb "#008080"
set object 50 rect from 0.108143, 30 to 146.089300, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.244337, 40 to 162.518761, 50 fc rgb "#FF0000"
set object 52 rect from 0.152577, 40 to 93.384174, 50 fc rgb "#00FF00"
set object 53 rect from 0.154013, 40 to 93.670323, 50 fc rgb "#0000FF"
set object 54 rect from 0.154281, 40 to 95.254764, 50 fc rgb "#FFFF00"
set object 55 rect from 0.156897, 40 to 99.966770, 50 fc rgb "#FF00FF"
set object 56 rect from 0.164647, 40 to 107.384118, 50 fc rgb "#808080"
set object 57 rect from 0.176857, 40 to 107.832472, 50 fc rgb "#800080"
set object 58 rect from 0.177812, 40 to 108.149605, 50 fc rgb "#008080"
set object 59 rect from 0.178118, 40 to 148.176775, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.239204, 50 to 169.273888, 60 fc rgb "#FF0000"
set object 61 rect from 0.027259, 50 to 18.307941, 60 fc rgb "#00FF00"
set object 62 rect from 0.030558, 50 to 18.908181, 60 fc rgb "#0000FF"
set object 63 rect from 0.031240, 50 to 21.302464, 60 fc rgb "#FFFF00"
set object 64 rect from 0.035202, 50 to 34.632905, 60 fc rgb "#FF00FF"
set object 65 rect from 0.057129, 50 to 42.229468, 60 fc rgb "#808080"
set object 66 rect from 0.069635, 50 to 42.752555, 60 fc rgb "#800080"
set object 67 rect from 0.070860, 50 to 43.550850, 60 fc rgb "#008080"
set object 68 rect from 0.071799, 50 to 144.804981, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
