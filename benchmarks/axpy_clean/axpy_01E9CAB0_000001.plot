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

set object 15 rect from 0.094072, 0 to 162.230811, 10 fc rgb "#FF0000"
set object 16 rect from 0.077508, 0 to 132.843301, 10 fc rgb "#00FF00"
set object 17 rect from 0.079084, 0 to 133.774148, 10 fc rgb "#0000FF"
set object 18 rect from 0.079430, 0 to 134.699941, 10 fc rgb "#FFFF00"
set object 19 rect from 0.079982, 0 to 136.182215, 10 fc rgb "#FF00FF"
set object 20 rect from 0.080857, 0 to 136.753877, 10 fc rgb "#808080"
set object 21 rect from 0.081195, 0 to 137.470563, 10 fc rgb "#800080"
set object 22 rect from 0.081855, 0 to 138.229408, 10 fc rgb "#008080"
set object 23 rect from 0.082073, 0 to 157.805909, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.092752, 10 to 160.514141, 20 fc rgb "#FF0000"
set object 25 rect from 0.065838, 10 to 113.212837, 20 fc rgb "#00FF00"
set object 26 rect from 0.067445, 10 to 114.199337, 20 fc rgb "#0000FF"
set object 27 rect from 0.067822, 10 to 115.265090, 20 fc rgb "#FFFF00"
set object 28 rect from 0.068482, 10 to 116.922743, 20 fc rgb "#FF00FF"
set object 29 rect from 0.069475, 10 to 117.609077, 20 fc rgb "#808080"
set object 30 rect from 0.069855, 10 to 118.474161, 20 fc rgb "#800080"
set object 31 rect from 0.070590, 10 to 119.340928, 20 fc rgb "#008080"
set object 32 rect from 0.070873, 10 to 155.558043, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.091445, 20 to 157.993091, 30 fc rgb "#FF0000"
set object 34 rect from 0.055511, 20 to 95.504783, 30 fc rgb "#00FF00"
set object 35 rect from 0.056965, 20 to 96.454181, 30 fc rgb "#0000FF"
set object 36 rect from 0.057298, 20 to 97.582330, 30 fc rgb "#FFFF00"
set object 37 rect from 0.057968, 20 to 99.071352, 30 fc rgb "#FF00FF"
set object 38 rect from 0.058854, 20 to 99.610975, 30 fc rgb "#808080"
set object 39 rect from 0.059170, 20 to 100.361386, 30 fc rgb "#800080"
set object 40 rect from 0.059849, 20 to 101.202862, 30 fc rgb "#008080"
set object 41 rect from 0.060115, 20 to 153.399551, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.090072, 30 to 155.871698, 40 fc rgb "#FF0000"
set object 43 rect from 0.044168, 30 to 76.597752, 40 fc rgb "#00FF00"
set object 44 rect from 0.045736, 30 to 77.444285, 40 fc rgb "#0000FF"
set object 45 rect from 0.046048, 30 to 78.702281, 40 fc rgb "#FFFF00"
set object 46 rect from 0.046771, 30 to 80.258756, 40 fc rgb "#FF00FF"
set object 47 rect from 0.047692, 30 to 80.811870, 40 fc rgb "#808080"
set object 48 rect from 0.048032, 30 to 81.666833, 40 fc rgb "#800080"
set object 49 rect from 0.048796, 30 to 82.557211, 40 fc rgb "#008080"
set object 50 rect from 0.049063, 30 to 150.989800, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.088650, 40 to 153.438336, 50 fc rgb "#FF0000"
set object 52 rect from 0.033010, 40 to 58.188189, 50 fc rgb "#00FF00"
set object 53 rect from 0.034813, 40 to 59.053270, 50 fc rgb "#0000FF"
set object 54 rect from 0.035119, 40 to 60.139261, 50 fc rgb "#FFFF00"
set object 55 rect from 0.035782, 40 to 61.894722, 50 fc rgb "#FF00FF"
set object 56 rect from 0.036813, 40 to 62.471444, 50 fc rgb "#808080"
set object 57 rect from 0.037165, 40 to 63.237033, 50 fc rgb "#800080"
set object 58 rect from 0.037839, 40 to 64.203294, 50 fc rgb "#008080"
set object 59 rect from 0.038191, 40 to 148.586790, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.087047, 50 to 154.171887, 60 fc rgb "#FF0000"
set object 61 rect from 0.017094, 50 to 32.679211, 60 fc rgb "#00FF00"
set object 62 rect from 0.019742, 50 to 34.223882, 60 fc rgb "#0000FF"
set object 63 rect from 0.020415, 50 to 36.862974, 60 fc rgb "#FFFF00"
set object 64 rect from 0.021995, 50 to 39.262608, 60 fc rgb "#FF00FF"
set object 65 rect from 0.023396, 50 to 40.217066, 60 fc rgb "#808080"
set object 66 rect from 0.023975, 50 to 41.316547, 60 fc rgb "#800080"
set object 67 rect from 0.024992, 50 to 42.780274, 60 fc rgb "#008080"
set object 68 rect from 0.025475, 50 to 145.391213, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
