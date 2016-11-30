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

set object 15 rect from 0.117837, 0 to 161.373903, 10 fc rgb "#FF0000"
set object 16 rect from 0.097586, 0 to 133.178361, 10 fc rgb "#00FF00"
set object 17 rect from 0.099779, 0 to 134.002245, 10 fc rgb "#0000FF"
set object 18 rect from 0.100152, 0 to 135.017687, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100911, 0 to 136.400195, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101930, 0 to 136.960177, 10 fc rgb "#808080"
set object 21 rect from 0.102376, 0 to 137.703653, 10 fc rgb "#800080"
set object 22 rect from 0.103161, 0 to 138.411037, 10 fc rgb "#008080"
set object 23 rect from 0.103435, 0 to 157.025396, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.116089, 10 to 159.641655, 20 fc rgb "#FF0000"
set object 25 rect from 0.083386, 10 to 114.167390, 20 fc rgb "#00FF00"
set object 26 rect from 0.085586, 10 to 115.102504, 20 fc rgb "#0000FF"
set object 27 rect from 0.086032, 10 to 116.163460, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086865, 10 to 117.823285, 20 fc rgb "#FF00FF"
set object 29 rect from 0.088116, 10 to 118.522604, 20 fc rgb "#808080"
set object 30 rect from 0.088619, 10 to 119.362537, 20 fc rgb "#800080"
set object 31 rect from 0.089470, 10 to 120.142265, 20 fc rgb "#008080"
set object 32 rect from 0.089803, 10 to 154.618024, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.114361, 20 to 156.896839, 30 fc rgb "#FF0000"
set object 34 rect from 0.070255, 20 to 96.461315, 30 fc rgb "#00FF00"
set object 35 rect from 0.072366, 20 to 97.257012, 30 fc rgb "#0000FF"
set object 36 rect from 0.072711, 20 to 98.393106, 30 fc rgb "#FFFF00"
set object 37 rect from 0.073558, 20 to 99.813064, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074622, 20 to 100.417283, 30 fc rgb "#808080"
set object 39 rect from 0.075092, 20 to 101.120594, 30 fc rgb "#800080"
set object 40 rect from 0.075880, 20 to 101.941763, 30 fc rgb "#008080"
set object 41 rect from 0.076233, 20 to 152.490682, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.112785, 30 to 154.897975, 40 fc rgb "#FF0000"
set object 43 rect from 0.054901, 30 to 75.956686, 40 fc rgb "#00FF00"
set object 44 rect from 0.057099, 30 to 76.807320, 40 fc rgb "#0000FF"
set object 45 rect from 0.057447, 30 to 78.010327, 40 fc rgb "#FFFF00"
set object 46 rect from 0.058354, 30 to 79.469251, 40 fc rgb "#FF00FF"
set object 47 rect from 0.059437, 30 to 80.025161, 40 fc rgb "#808080"
set object 48 rect from 0.059863, 30 to 80.783409, 40 fc rgb "#800080"
set object 49 rect from 0.060669, 30 to 81.567129, 40 fc rgb "#008080"
set object 50 rect from 0.061005, 30 to 150.343218, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.111185, 40 to 152.839064, 50 fc rgb "#FF0000"
set object 52 rect from 0.040753, 40 to 57.279326, 50 fc rgb "#00FF00"
set object 53 rect from 0.043134, 40 to 58.139381, 50 fc rgb "#0000FF"
set object 54 rect from 0.043514, 40 to 59.278030, 50 fc rgb "#FFFF00"
set object 55 rect from 0.044384, 40 to 60.864234, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045560, 40 to 61.459031, 50 fc rgb "#808080"
set object 57 rect from 0.045995, 40 to 62.189092, 50 fc rgb "#800080"
set object 58 rect from 0.046802, 40 to 63.035733, 50 fc rgb "#008080"
set object 59 rect from 0.047196, 40 to 148.169005, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.109477, 50 to 154.113058, 60 fc rgb "#FF0000"
set object 61 rect from 0.019507, 50 to 30.162147, 60 fc rgb "#00FF00"
set object 62 rect from 0.022974, 50 to 31.698765, 60 fc rgb "#0000FF"
set object 63 rect from 0.023797, 50 to 34.544190, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026052, 50 to 37.049298, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027790, 50 to 38.000461, 60 fc rgb "#808080"
set object 66 rect from 0.028504, 50 to 38.959610, 60 fc rgb "#800080"
set object 67 rect from 0.029623, 50 to 40.268497, 60 fc rgb "#008080"
set object 68 rect from 0.030196, 50 to 145.405266, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
