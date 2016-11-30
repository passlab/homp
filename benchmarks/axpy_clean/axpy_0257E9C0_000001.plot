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

set object 15 rect from 0.128565, 0 to 153.265136, 10 fc rgb "#FF0000"
set object 16 rect from 0.029369, 0 to 36.899906, 10 fc rgb "#00FF00"
set object 17 rect from 0.032839, 0 to 38.129409, 10 fc rgb "#0000FF"
set object 18 rect from 0.033604, 0 to 40.008418, 10 fc rgb "#FFFF00"
set object 19 rect from 0.035274, 0 to 42.150649, 10 fc rgb "#FF00FF"
set object 20 rect from 0.037129, 0 to 43.921404, 10 fc rgb "#808080"
set object 21 rect from 0.038706, 0 to 44.728162, 10 fc rgb "#800080"
set object 22 rect from 0.039830, 0 to 45.677353, 10 fc rgb "#008080"
set object 23 rect from 0.040222, 0 to 145.262546, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.137031, 10 to 160.171547, 20 fc rgb "#FF0000"
set object 25 rect from 0.114477, 10 to 132.712271, 20 fc rgb "#00FF00"
set object 26 rect from 0.116822, 10 to 133.425589, 20 fc rgb "#0000FF"
set object 27 rect from 0.117233, 10 to 134.302991, 20 fc rgb "#FFFF00"
set object 28 rect from 0.117987, 10 to 135.483496, 20 fc rgb "#FF00FF"
set object 29 rect from 0.119031, 10 to 136.948872, 20 fc rgb "#808080"
set object 30 rect from 0.120333, 10 to 137.639400, 20 fc rgb "#800080"
set object 31 rect from 0.121156, 10 to 138.246744, 20 fc rgb "#008080"
set object 32 rect from 0.121456, 10 to 155.490555, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.133855, 20 to 156.733736, 30 fc rgb "#FF0000"
set object 34 rect from 0.085013, 20 to 98.870737, 30 fc rgb "#00FF00"
set object 35 rect from 0.087150, 20 to 99.576079, 30 fc rgb "#0000FF"
set object 36 rect from 0.087500, 20 to 100.598198, 30 fc rgb "#FFFF00"
set object 37 rect from 0.088397, 20 to 101.847071, 30 fc rgb "#FF00FF"
set object 38 rect from 0.089495, 20 to 103.187104, 30 fc rgb "#808080"
set object 39 rect from 0.090671, 20 to 103.908399, 30 fc rgb "#800080"
set object 40 rect from 0.091560, 20 to 104.661594, 30 fc rgb "#008080"
set object 41 rect from 0.091966, 20 to 151.938785, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.135343, 30 to 158.275448, 40 fc rgb "#FF0000"
set object 43 rect from 0.099786, 30 to 115.502644, 40 fc rgb "#00FF00"
set object 44 rect from 0.101716, 30 to 116.234195, 40 fc rgb "#0000FF"
set object 45 rect from 0.102121, 30 to 117.187941, 40 fc rgb "#FFFF00"
set object 46 rect from 0.102957, 30 to 118.406051, 40 fc rgb "#FF00FF"
set object 47 rect from 0.104023, 30 to 119.715317, 40 fc rgb "#808080"
set object 48 rect from 0.105177, 30 to 120.319244, 40 fc rgb "#800080"
set object 49 rect from 0.105965, 30 to 121.021165, 40 fc rgb "#008080"
set object 50 rect from 0.106324, 30 to 153.693588, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.130496, 40 to 153.820078, 50 fc rgb "#FF0000"
set object 52 rect from 0.051104, 40 to 60.847301, 50 fc rgb "#00FF00"
set object 53 rect from 0.053761, 40 to 61.697357, 50 fc rgb "#0000FF"
set object 54 rect from 0.054276, 40 to 62.973576, 50 fc rgb "#FFFF00"
set object 55 rect from 0.055421, 40 to 64.652038, 50 fc rgb "#FF00FF"
set object 56 rect from 0.056887, 40 to 66.103742, 50 fc rgb "#808080"
set object 57 rect from 0.058135, 40 to 66.894543, 50 fc rgb "#800080"
set object 58 rect from 0.059097, 40 to 68.133160, 50 fc rgb "#008080"
set object 59 rect from 0.059947, 40 to 147.819547, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.132233, 50 to 154.849026, 60 fc rgb "#FF0000"
set object 61 rect from 0.068834, 50 to 80.646972, 60 fc rgb "#00FF00"
set object 62 rect from 0.071124, 50 to 81.328386, 60 fc rgb "#0000FF"
set object 63 rect from 0.071486, 50 to 82.310619, 60 fc rgb "#FFFF00"
set object 64 rect from 0.072345, 50 to 83.566331, 60 fc rgb "#FF00FF"
set object 65 rect from 0.073451, 50 to 84.902946, 60 fc rgb "#808080"
set object 66 rect from 0.074639, 50 to 85.616261, 60 fc rgb "#800080"
set object 67 rect from 0.075503, 50 to 86.313627, 60 fc rgb "#008080"
set object 68 rect from 0.075862, 50 to 149.899107, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
