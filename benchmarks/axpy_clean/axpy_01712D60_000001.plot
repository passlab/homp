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

set object 15 rect from 0.185457, 0 to 161.659525, 10 fc rgb "#FF0000"
set object 16 rect from 0.157395, 0 to 130.637167, 10 fc rgb "#00FF00"
set object 17 rect from 0.159264, 0 to 131.147743, 10 fc rgb "#0000FF"
set object 18 rect from 0.159633, 0 to 131.741359, 10 fc rgb "#FFFF00"
set object 19 rect from 0.160354, 0 to 132.582441, 10 fc rgb "#FF00FF"
set object 20 rect from 0.161399, 0 to 139.982027, 10 fc rgb "#808080"
set object 21 rect from 0.170406, 0 to 140.460529, 10 fc rgb "#800080"
set object 22 rect from 0.171230, 0 to 140.925885, 10 fc rgb "#008080"
set object 23 rect from 0.171547, 0 to 152.002231, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.183849, 10 to 160.767441, 20 fc rgb "#FF0000"
set object 25 rect from 0.134709, 10 to 112.069154, 20 fc rgb "#00FF00"
set object 26 rect from 0.136675, 10 to 112.603559, 20 fc rgb "#0000FF"
set object 27 rect from 0.137078, 10 to 113.209512, 20 fc rgb "#FFFF00"
set object 28 rect from 0.137836, 10 to 114.175569, 20 fc rgb "#FF00FF"
set object 29 rect from 0.139002, 10 to 121.820158, 20 fc rgb "#808080"
set object 30 rect from 0.148333, 10 to 122.341418, 20 fc rgb "#800080"
set object 31 rect from 0.149201, 10 to 122.852815, 20 fc rgb "#008080"
set object 32 rect from 0.149550, 10 to 150.657973, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.182193, 20 to 164.140012, 30 fc rgb "#FF0000"
set object 34 rect from 0.107561, 20 to 89.576890, 30 fc rgb "#00FF00"
set object 35 rect from 0.109350, 20 to 90.072654, 30 fc rgb "#0000FF"
set object 36 rect from 0.109674, 20 to 92.618110, 30 fc rgb "#FFFF00"
set object 37 rect from 0.112772, 20 to 96.627863, 30 fc rgb "#FF00FF"
set object 38 rect from 0.117650, 20 to 104.011828, 30 fc rgb "#808080"
set object 39 rect from 0.126646, 20 to 104.565149, 30 fc rgb "#800080"
set object 40 rect from 0.127611, 20 to 105.112724, 30 fc rgb "#008080"
set object 41 rect from 0.127987, 20 to 149.396756, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.180707, 30 to 162.902648, 40 fc rgb "#FF0000"
set object 43 rect from 0.078261, 30 to 65.636770, 40 fc rgb "#00FF00"
set object 44 rect from 0.080220, 30 to 66.134176, 40 fc rgb "#0000FF"
set object 45 rect from 0.080563, 30 to 68.765968, 40 fc rgb "#FFFF00"
set object 46 rect from 0.083781, 30 to 72.699260, 40 fc rgb "#FF00FF"
set object 47 rect from 0.088570, 30 to 80.061025, 40 fc rgb "#808080"
set object 48 rect from 0.097531, 30 to 80.652167, 40 fc rgb "#800080"
set object 49 rect from 0.098507, 30 to 81.215350, 40 fc rgb "#008080"
set object 50 rect from 0.098929, 30 to 148.137179, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.179153, 40 to 162.013872, 50 fc rgb "#FF0000"
set object 52 rect from 0.049651, 40 to 42.319854, 50 fc rgb "#00FF00"
set object 53 rect from 0.051841, 40 to 42.833713, 50 fc rgb "#0000FF"
set object 54 rect from 0.052219, 40 to 45.546070, 50 fc rgb "#FFFF00"
set object 55 rect from 0.055550, 40 to 49.768776, 50 fc rgb "#FF00FF"
set object 56 rect from 0.060672, 40 to 57.112447, 50 fc rgb "#808080"
set object 57 rect from 0.069601, 40 to 57.692084, 50 fc rgb "#800080"
set object 58 rect from 0.070596, 40 to 58.310362, 50 fc rgb "#008080"
set object 59 rect from 0.071072, 40 to 146.734556, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.177281, 50 to 162.178285, 60 fc rgb "#FF0000"
set object 61 rect from 0.016023, 50 to 15.373859, 60 fc rgb "#00FF00"
set object 62 rect from 0.019180, 50 to 16.279077, 60 fc rgb "#0000FF"
set object 63 rect from 0.019933, 50 to 19.933650, 60 fc rgb "#FFFF00"
set object 64 rect from 0.024437, 50 to 24.289541, 60 fc rgb "#FF00FF"
set object 65 rect from 0.029697, 50 to 31.861773, 60 fc rgb "#808080"
set object 66 rect from 0.038902, 50 to 32.598448, 60 fc rgb "#800080"
set object 67 rect from 0.040213, 50 to 33.585885, 60 fc rgb "#008080"
set object 68 rect from 0.040979, 50 to 144.915901, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
