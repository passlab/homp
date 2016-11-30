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

set object 15 rect from 0.116015, 0 to 162.150052, 10 fc rgb "#FF0000"
set object 16 rect from 0.095456, 0 to 132.964059, 10 fc rgb "#00FF00"
set object 17 rect from 0.097668, 0 to 133.934295, 10 fc rgb "#0000FF"
set object 18 rect from 0.098124, 0 to 134.904530, 10 fc rgb "#FFFF00"
set object 19 rect from 0.098835, 0 to 136.275094, 10 fc rgb "#FF00FF"
set object 20 rect from 0.099837, 0 to 136.843535, 10 fc rgb "#808080"
set object 21 rect from 0.100281, 0 to 137.576017, 10 fc rgb "#800080"
set object 22 rect from 0.101047, 0 to 138.274291, 10 fc rgb "#008080"
set object 23 rect from 0.101302, 0 to 157.706044, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.114384, 10 to 160.398057, 20 fc rgb "#FF0000"
set object 25 rect from 0.080984, 10 to 113.166105, 20 fc rgb "#00FF00"
set object 26 rect from 0.083166, 10 to 114.065235, 20 fc rgb "#0000FF"
set object 27 rect from 0.083586, 10 to 115.125636, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084396, 10 to 116.738107, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085565, 10 to 117.484273, 20 fc rgb "#808080"
set object 30 rect from 0.086121, 10 to 118.330134, 20 fc rgb "#800080"
set object 31 rect from 0.086963, 10 to 119.122726, 20 fc rgb "#008080"
set object 32 rect from 0.087295, 10 to 155.404829, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.112696, 20 to 157.615960, 30 fc rgb "#FF0000"
set object 34 rect from 0.068489, 20 to 96.080609, 30 fc rgb "#00FF00"
set object 35 rect from 0.070682, 20 to 96.880043, 30 fc rgb "#0000FF"
set object 36 rect from 0.071007, 20 to 97.989640, 30 fc rgb "#FFFF00"
set object 37 rect from 0.071821, 20 to 99.423083, 30 fc rgb "#FF00FF"
set object 38 rect from 0.072878, 20 to 99.990221, 30 fc rgb "#808080"
set object 39 rect from 0.073286, 20 to 100.677580, 30 fc rgb "#800080"
set object 40 rect from 0.074043, 20 to 101.482390, 30 fc rgb "#008080"
set object 41 rect from 0.074399, 20 to 153.344138, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.111110, 30 to 155.788704, 40 fc rgb "#FF0000"
set object 43 rect from 0.053799, 30 to 76.230690, 40 fc rgb "#00FF00"
set object 44 rect from 0.056140, 30 to 77.103918, 40 fc rgb "#0000FF"
set object 45 rect from 0.056539, 30 to 78.377474, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057475, 30 to 79.830059, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058539, 30 to 80.416338, 40 fc rgb "#808080"
set object 48 rect from 0.058971, 30 to 81.173336, 40 fc rgb "#800080"
set object 49 rect from 0.059798, 30 to 82.030193, 40 fc rgb "#008080"
set object 50 rect from 0.060152, 30 to 151.049765, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.109455, 40 to 153.647621, 50 fc rgb "#FF0000"
set object 52 rect from 0.039656, 40 to 57.024472, 50 fc rgb "#00FF00"
set object 53 rect from 0.042104, 40 to 57.942743, 50 fc rgb "#0000FF"
set object 54 rect from 0.042547, 40 to 59.153419, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043405, 40 to 60.734532, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044577, 40 to 61.364468, 50 fc rgb "#808080"
set object 57 rect from 0.045030, 40 to 62.161134, 50 fc rgb "#800080"
set object 58 rect from 0.045908, 40 to 63.052118, 50 fc rgb "#008080"
set object 59 rect from 0.046277, 40 to 148.760931, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.107567, 50 to 156.138778, 60 fc rgb "#FF0000"
set object 61 rect from 0.019317, 50 to 30.063212, 60 fc rgb "#00FF00"
set object 62 rect from 0.022467, 50 to 31.801442, 60 fc rgb "#0000FF"
set object 63 rect from 0.023412, 50 to 35.726042, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026317, 50 to 38.065457, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027991, 50 to 39.377376, 60 fc rgb "#808080"
set object 66 rect from 0.028955, 50 to 40.459606, 60 fc rgb "#800080"
set object 67 rect from 0.030178, 50 to 41.852080, 60 fc rgb "#008080"
set object 68 rect from 0.030763, 50 to 145.698564, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
