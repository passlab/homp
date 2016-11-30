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

set object 15 rect from 0.145084, 0 to 157.243470, 10 fc rgb "#FF0000"
set object 16 rect from 0.109383, 0 to 117.587116, 10 fc rgb "#00FF00"
set object 17 rect from 0.111642, 0 to 118.365749, 10 fc rgb "#0000FF"
set object 18 rect from 0.112149, 0 to 119.210976, 10 fc rgb "#FFFF00"
set object 19 rect from 0.112956, 0 to 120.329823, 10 fc rgb "#FF00FF"
set object 20 rect from 0.114033, 0 to 121.714920, 10 fc rgb "#808080"
set object 21 rect from 0.115353, 0 to 122.329791, 10 fc rgb "#800080"
set object 22 rect from 0.116145, 0 to 122.903478, 10 fc rgb "#008080"
set object 23 rect from 0.116449, 0 to 152.671893, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138325, 10 to 154.230293, 20 fc rgb "#FF0000"
set object 25 rect from 0.038913, 10 to 44.516318, 20 fc rgb "#00FF00"
set object 26 rect from 0.042655, 10 to 46.088414, 20 fc rgb "#0000FF"
set object 27 rect from 0.043761, 10 to 49.191392, 20 fc rgb "#FFFF00"
set object 28 rect from 0.046771, 10 to 51.118500, 20 fc rgb "#FF00FF"
set object 29 rect from 0.048528, 10 to 52.957887, 20 fc rgb "#808080"
set object 30 rect from 0.050271, 10 to 53.687936, 20 fc rgb "#800080"
set object 31 rect from 0.051473, 10 to 54.596546, 20 fc rgb "#008080"
set object 32 rect from 0.051827, 10 to 145.104079, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.143399, 20 to 155.268817, 30 fc rgb "#FF0000"
set object 34 rect from 0.094558, 20 to 101.689699, 30 fc rgb "#00FF00"
set object 35 rect from 0.096630, 20 to 102.412349, 30 fc rgb "#0000FF"
set object 36 rect from 0.097045, 20 to 103.236449, 30 fc rgb "#FFFF00"
set object 37 rect from 0.097828, 20 to 104.444025, 30 fc rgb "#FF00FF"
set object 38 rect from 0.098969, 20 to 105.640013, 30 fc rgb "#808080"
set object 39 rect from 0.100102, 20 to 106.212661, 30 fc rgb "#800080"
set object 40 rect from 0.100952, 20 to 106.902565, 30 fc rgb "#008080"
set object 41 rect from 0.101306, 20 to 150.906469, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.141828, 30 to 153.867851, 40 fc rgb "#FF0000"
set object 43 rect from 0.079237, 30 to 85.662337, 40 fc rgb "#00FF00"
set object 44 rect from 0.081452, 30 to 86.390277, 40 fc rgb "#0000FF"
set object 45 rect from 0.081902, 30 to 87.350651, 40 fc rgb "#FFFF00"
set object 46 rect from 0.082793, 30 to 88.529731, 40 fc rgb "#FF00FF"
set object 47 rect from 0.083919, 30 to 89.762685, 40 fc rgb "#808080"
set object 48 rect from 0.085075, 30 to 90.475826, 40 fc rgb "#800080"
set object 49 rect from 0.086044, 30 to 91.303075, 40 fc rgb "#008080"
set object 50 rect from 0.086539, 30 to 149.189628, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.140212, 40 to 153.238181, 50 fc rgb "#FF0000"
set object 52 rect from 0.062204, 40 to 68.010076, 50 fc rgb "#00FF00"
set object 53 rect from 0.064731, 40 to 68.915506, 50 fc rgb "#0000FF"
set object 54 rect from 0.065359, 40 to 70.192824, 50 fc rgb "#FFFF00"
set object 55 rect from 0.066582, 40 to 71.945591, 50 fc rgb "#FF00FF"
set object 56 rect from 0.068233, 40 to 73.200743, 50 fc rgb "#808080"
set object 57 rect from 0.069433, 40 to 73.962468, 50 fc rgb "#800080"
set object 58 rect from 0.070418, 40 to 75.143658, 50 fc rgb "#008080"
set object 59 rect from 0.071271, 40 to 147.486516, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.146567, 50 to 158.702498, 60 fc rgb "#FF0000"
set object 61 rect from 0.124844, 50 to 133.781388, 60 fc rgb "#00FF00"
set object 62 rect from 0.126993, 50 to 134.432186, 60 fc rgb "#0000FF"
set object 63 rect from 0.127360, 50 to 135.303830, 60 fc rgb "#FFFF00"
set object 64 rect from 0.128196, 50 to 136.486059, 60 fc rgb "#FF00FF"
set object 65 rect from 0.129313, 50 to 137.764448, 60 fc rgb "#808080"
set object 66 rect from 0.130522, 50 to 138.436405, 60 fc rgb "#800080"
set object 67 rect from 0.131427, 50 to 139.113620, 60 fc rgb "#008080"
set object 68 rect from 0.131791, 50 to 154.346479, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
