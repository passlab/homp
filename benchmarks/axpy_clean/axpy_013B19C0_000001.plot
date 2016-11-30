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

set object 15 rect from 0.143350, 0 to 153.474911, 10 fc rgb "#FF0000"
set object 16 rect from 0.080033, 0 to 85.562523, 10 fc rgb "#00FF00"
set object 17 rect from 0.082263, 0 to 86.313708, 10 fc rgb "#0000FF"
set object 18 rect from 0.082735, 0 to 87.105634, 10 fc rgb "#FFFF00"
set object 19 rect from 0.083508, 0 to 88.270533, 10 fc rgb "#FF00FF"
set object 20 rect from 0.084629, 0 to 89.481412, 10 fc rgb "#808080"
set object 21 rect from 0.085788, 0 to 90.085283, 10 fc rgb "#800080"
set object 22 rect from 0.086626, 0 to 90.690197, 10 fc rgb "#008080"
set object 23 rect from 0.086947, 0 to 149.097379, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.139848, 10 to 152.873126, 20 fc rgb "#FF0000"
set object 25 rect from 0.041308, 10 to 46.672459, 20 fc rgb "#00FF00"
set object 26 rect from 0.045129, 10 to 48.117358, 20 fc rgb "#0000FF"
set object 27 rect from 0.046189, 10 to 50.424185, 20 fc rgb "#FFFF00"
set object 28 rect from 0.048425, 10 to 52.008037, 20 fc rgb "#FF00FF"
set object 29 rect from 0.049916, 10 to 53.679654, 20 fc rgb "#808080"
set object 30 rect from 0.051534, 10 to 54.386952, 20 fc rgb "#800080"
set object 31 rect from 0.052628, 10 to 55.252009, 20 fc rgb "#008080"
set object 32 rect from 0.053024, 10 to 145.154461, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.144976, 20 to 155.406664, 30 fc rgb "#FF0000"
set object 34 rect from 0.095617, 20 to 101.652836, 30 fc rgb "#00FF00"
set object 35 rect from 0.097674, 20 to 102.359090, 30 fc rgb "#0000FF"
set object 36 rect from 0.098096, 20 to 103.294150, 30 fc rgb "#FFFF00"
set object 37 rect from 0.099033, 20 to 104.799647, 30 fc rgb "#FF00FF"
set object 38 rect from 0.100451, 20 to 105.810975, 30 fc rgb "#808080"
set object 39 rect from 0.101408, 20 to 106.446185, 30 fc rgb "#800080"
set object 40 rect from 0.102276, 20 to 107.206772, 30 fc rgb "#008080"
set object 41 rect from 0.102736, 20 to 150.841076, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.141696, 30 to 152.674617, 40 fc rgb "#FF0000"
set object 43 rect from 0.063135, 30 to 68.056564, 40 fc rgb "#00FF00"
set object 44 rect from 0.065504, 30 to 68.926845, 40 fc rgb "#0000FF"
set object 45 rect from 0.066095, 30 to 70.032203, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067208, 30 to 71.769635, 40 fc rgb "#FF00FF"
set object 47 rect from 0.068856, 30 to 72.825884, 40 fc rgb "#808080"
set object 48 rect from 0.069830, 30 to 73.530045, 40 fc rgb "#800080"
set object 49 rect from 0.070789, 30 to 74.638533, 40 fc rgb "#008080"
set object 50 rect from 0.071591, 30 to 147.312931, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.147855, 40 to 157.872293, 50 fc rgb "#FF0000"
set object 52 rect from 0.126504, 40 to 133.634959, 50 fc rgb "#00FF00"
set object 53 rect from 0.128278, 40 to 134.390323, 50 fc rgb "#0000FF"
set object 54 rect from 0.128753, 40 to 135.161349, 50 fc rgb "#FFFF00"
set object 55 rect from 0.129491, 40 to 136.298047, 50 fc rgb "#FF00FF"
set object 56 rect from 0.130576, 40 to 137.220573, 50 fc rgb "#808080"
set object 57 rect from 0.131462, 40 to 137.794141, 50 fc rgb "#800080"
set object 58 rect from 0.132270, 40 to 138.412639, 50 fc rgb "#008080"
set object 59 rect from 0.132606, 40 to 153.987888, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.146400, 50 to 156.516210, 60 fc rgb "#FF0000"
set object 61 rect from 0.110751, 50 to 117.173760, 60 fc rgb "#00FF00"
set object 62 rect from 0.112536, 50 to 117.874792, 60 fc rgb "#0000FF"
set object 63 rect from 0.112945, 50 to 118.709553, 60 fc rgb "#FFFF00"
set object 64 rect from 0.113777, 50 to 119.989384, 60 fc rgb "#FF00FF"
set object 65 rect from 0.114965, 50 to 120.968322, 60 fc rgb "#808080"
set object 66 rect from 0.115905, 50 to 121.543984, 60 fc rgb "#800080"
set object 67 rect from 0.116755, 50 to 122.210540, 60 fc rgb "#008080"
set object 68 rect from 0.117099, 50 to 152.488656, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
