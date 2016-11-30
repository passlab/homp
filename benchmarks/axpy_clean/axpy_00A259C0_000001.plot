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

set object 15 rect from 0.148767, 0 to 152.309575, 10 fc rgb "#FF0000"
set object 16 rect from 0.048788, 0 to 50.960721, 10 fc rgb "#00FF00"
set object 17 rect from 0.051463, 0 to 51.688931, 10 fc rgb "#0000FF"
set object 18 rect from 0.051928, 0 to 52.513903, 10 fc rgb "#FFFF00"
set object 19 rect from 0.052789, 0 to 53.755849, 10 fc rgb "#FF00FF"
set object 20 rect from 0.054044, 0 to 55.085581, 10 fc rgb "#808080"
set object 21 rect from 0.055389, 0 to 55.709049, 10 fc rgb "#800080"
set object 22 rect from 0.056229, 0 to 56.296603, 10 fc rgb "#008080"
set object 23 rect from 0.056565, 0 to 147.703895, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.155991, 10 to 159.296395, 20 fc rgb "#FF0000"
set object 25 rect from 0.131746, 10 to 133.381104, 20 fc rgb "#00FF00"
set object 26 rect from 0.134073, 10 to 134.049462, 20 fc rgb "#0000FF"
set object 27 rect from 0.134495, 10 to 134.825553, 20 fc rgb "#FFFF00"
set object 28 rect from 0.135278, 10 to 135.892930, 20 fc rgb "#FF00FF"
set object 29 rect from 0.136354, 10 to 137.229644, 20 fc rgb "#808080"
set object 30 rect from 0.137699, 10 to 137.813210, 20 fc rgb "#800080"
set object 31 rect from 0.138547, 10 to 138.375826, 20 fc rgb "#008080"
set object 32 rect from 0.138854, 10 to 154.992977, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.146710, 20 to 152.717570, 30 fc rgb "#FF0000"
set object 34 rect from 0.027056, 20 to 29.913463, 30 fc rgb "#00FF00"
set object 35 rect from 0.030515, 20 to 31.122490, 30 fc rgb "#0000FF"
set object 36 rect from 0.031332, 20 to 33.263227, 30 fc rgb "#FFFF00"
set object 37 rect from 0.033510, 20 to 35.004946, 30 fc rgb "#FF00FF"
set object 38 rect from 0.035223, 20 to 36.492289, 30 fc rgb "#808080"
set object 39 rect from 0.036741, 20 to 37.260401, 30 fc rgb "#800080"
set object 40 rect from 0.037878, 20 to 38.359699, 30 fc rgb "#008080"
set object 41 rect from 0.038589, 20 to 145.278857, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.151035, 30 to 154.516151, 40 fc rgb "#FF0000"
set object 43 rect from 0.065240, 30 to 67.098051, 40 fc rgb "#00FF00"
set object 44 rect from 0.067715, 30 to 67.869155, 40 fc rgb "#0000FF"
set object 45 rect from 0.068148, 30 to 68.738020, 40 fc rgb "#FFFF00"
set object 46 rect from 0.069049, 30 to 69.903155, 40 fc rgb "#FF00FF"
set object 47 rect from 0.070200, 30 to 71.124153, 40 fc rgb "#808080"
set object 48 rect from 0.071427, 30 to 71.793507, 40 fc rgb "#800080"
set object 49 rect from 0.072358, 30 to 72.534685, 40 fc rgb "#008080"
set object 50 rect from 0.072868, 30 to 149.776800, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.154154, 40 to 157.717277, 50 fc rgb "#FF0000"
set object 52 rect from 0.114660, 40 to 116.184381, 50 fc rgb "#00FF00"
set object 53 rect from 0.116832, 40 to 116.810841, 50 fc rgb "#0000FF"
set object 54 rect from 0.117215, 40 to 117.718608, 50 fc rgb "#FFFF00"
set object 55 rect from 0.118127, 40 to 119.046344, 50 fc rgb "#FF00FF"
set object 56 rect from 0.119452, 40 to 120.238414, 50 fc rgb "#808080"
set object 57 rect from 0.120651, 40 to 120.818987, 50 fc rgb "#800080"
set object 58 rect from 0.121502, 40 to 121.442454, 50 fc rgb "#008080"
set object 59 rect from 0.121861, 40 to 153.308119, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.152651, 50 to 156.481317, 60 fc rgb "#FF0000"
set object 61 rect from 0.096986, 50 to 98.902864, 60 fc rgb "#00FF00"
set object 62 rect from 0.099513, 50 to 99.641048, 60 fc rgb "#0000FF"
set object 63 rect from 0.100000, 50 to 100.682489, 60 fc rgb "#FFFF00"
set object 64 rect from 0.101043, 50 to 101.930421, 60 fc rgb "#FF00FF"
set object 65 rect from 0.102296, 50 to 103.172369, 60 fc rgb "#808080"
set object 66 rect from 0.103562, 50 to 103.836736, 60 fc rgb "#800080"
set object 67 rect from 0.104476, 50 to 104.731535, 60 fc rgb "#008080"
set object 68 rect from 0.105105, 50 to 151.695083, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
