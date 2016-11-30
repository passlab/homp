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

set object 15 rect from 0.351545, 0 to 154.781151, 10 fc rgb "#FF0000"
set object 16 rect from 0.280212, 0 to 118.374004, 10 fc rgb "#00FF00"
set object 17 rect from 0.281815, 0 to 118.581730, 10 fc rgb "#0000FF"
set object 18 rect from 0.282103, 0 to 118.813004, 10 fc rgb "#FFFF00"
set object 19 rect from 0.282670, 0 to 119.176315, 10 fc rgb "#FF00FF"
set object 20 rect from 0.283549, 0 to 125.418188, 10 fc rgb "#808080"
set object 21 rect from 0.298368, 0 to 125.609095, 10 fc rgb "#800080"
set object 22 rect from 0.299069, 0 to 125.807990, 10 fc rgb "#008080"
set object 23 rect from 0.299294, 0 to 147.615863, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.350063, 10 to 154.412375, 20 fc rgb "#FF0000"
set object 25 rect from 0.253178, 10 to 107.057979, 20 fc rgb "#00FF00"
set object 26 rect from 0.254926, 10 to 107.325837, 20 fc rgb "#0000FF"
set object 27 rect from 0.255351, 10 to 107.584443, 20 fc rgb "#FFFF00"
set object 28 rect from 0.255982, 10 to 108.018397, 20 fc rgb "#FF00FF"
set object 29 rect from 0.256997, 10 to 114.352780, 20 fc rgb "#808080"
set object 30 rect from 0.272063, 10 to 114.567655, 20 fc rgb "#800080"
set object 31 rect from 0.272818, 10 to 114.803554, 20 fc rgb "#008080"
set object 32 rect from 0.273134, 10 to 146.879992, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.348297, 20 to 161.127309, 30 fc rgb "#FF0000"
set object 34 rect from 0.206948, 20 to 87.605316, 30 fc rgb "#00FF00"
set object 35 rect from 0.208667, 20 to 87.882844, 30 fc rgb "#0000FF"
set object 36 rect from 0.209122, 20 to 91.101758, 30 fc rgb "#FFFF00"
set object 37 rect from 0.216830, 20 to 96.084661, 30 fc rgb "#FF00FF"
set object 38 rect from 0.228632, 20 to 102.265983, 30 fc rgb "#808080"
set object 39 rect from 0.243323, 20 to 102.604484, 30 fc rgb "#800080"
set object 40 rect from 0.244367, 20 to 102.850895, 30 fc rgb "#008080"
set object 41 rect from 0.244696, 20 to 146.206354, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.346600, 30 to 206.430936, 40 fc rgb "#FF0000"
set object 43 rect from 0.049356, 30 to 22.325086, 40 fc rgb "#00FF00"
set object 44 rect from 0.053637, 30 to 22.946162, 40 fc rgb "#0000FF"
set object 45 rect from 0.054713, 30 to 25.151674, 40 fc rgb "#FFFF00"
set object 46 rect from 0.060064, 30 to 31.433076, 40 fc rgb "#FF00FF"
set object 47 rect from 0.074894, 30 to 82.822571, 40 fc rgb "#808080"
set object 48 rect from 0.197199, 30 to 83.523120, 40 fc rgb "#800080"
set object 49 rect from 0.199001, 30 to 83.802752, 40 fc rgb "#008080"
set object 50 rect from 0.199439, 30 to 145.502019, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.352918, 40 to 160.464184, 50 fc rgb "#FF0000"
set object 52 rect from 0.306309, 40 to 129.325877, 50 fc rgb "#00FF00"
set object 53 rect from 0.307857, 40 to 129.560515, 50 fc rgb "#0000FF"
set object 54 rect from 0.308212, 40 to 130.642457, 50 fc rgb "#FFFF00"
set object 55 rect from 0.310803, 40 to 135.169119, 50 fc rgb "#FF00FF"
set object 56 rect from 0.321580, 40 to 141.338247, 50 fc rgb "#808080"
set object 57 rect from 0.336237, 40 to 141.669179, 50 fc rgb "#800080"
set object 58 rect from 0.337253, 40 to 141.927785, 50 fc rgb "#008080"
set object 59 rect from 0.337626, 40 to 148.212971, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.344897, 50 to 157.693100, 60 fc rgb "#FF0000"
set object 61 rect from 0.154816, 50 to 66.049742, 60 fc rgb "#00FF00"
set object 62 rect from 0.157407, 50 to 66.396652, 60 fc rgb "#0000FF"
set object 63 rect from 0.158000, 50 to 67.699776, 60 fc rgb "#FFFF00"
set object 64 rect from 0.161128, 50 to 72.368146, 60 fc rgb "#FF00FF"
set object 65 rect from 0.172219, 50 to 78.601190, 60 fc rgb "#808080"
set object 66 rect from 0.187065, 50 to 79.012017, 60 fc rgb "#800080"
set object 67 rect from 0.188375, 50 to 79.473303, 60 fc rgb "#008080"
set object 68 rect from 0.189114, 50 to 144.474742, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
