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

set object 15 rect from 0.142824, 0 to 157.354865, 10 fc rgb "#FF0000"
set object 16 rect from 0.107373, 0 to 117.282749, 10 fc rgb "#00FF00"
set object 17 rect from 0.109503, 0 to 117.986598, 10 fc rgb "#0000FF"
set object 18 rect from 0.109912, 0 to 118.827993, 10 fc rgb "#FFFF00"
set object 19 rect from 0.110700, 0 to 119.932659, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111724, 0 to 121.319942, 10 fc rgb "#808080"
set object 21 rect from 0.113014, 0 to 121.895916, 10 fc rgb "#800080"
set object 22 rect from 0.113796, 0 to 122.447174, 10 fc rgb "#008080"
set object 23 rect from 0.114067, 0 to 152.884618, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.141214, 10 to 156.026689, 20 fc rgb "#FF0000"
set object 25 rect from 0.092363, 10 to 101.411220, 20 fc rgb "#00FF00"
set object 26 rect from 0.094741, 10 to 102.391237, 20 fc rgb "#0000FF"
set object 27 rect from 0.095397, 10 to 103.234780, 20 fc rgb "#FFFF00"
set object 28 rect from 0.096225, 10 to 104.513530, 20 fc rgb "#FF00FF"
set object 29 rect from 0.097403, 10 to 105.900811, 20 fc rgb "#808080"
set object 30 rect from 0.098699, 10 to 106.544484, 20 fc rgb "#800080"
set object 31 rect from 0.099533, 10 to 107.197827, 20 fc rgb "#008080"
set object 32 rect from 0.099882, 10 to 151.044939, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.139466, 20 to 154.015078, 30 fc rgb "#FF0000"
set object 34 rect from 0.076055, 20 to 83.594706, 30 fc rgb "#00FF00"
set object 35 rect from 0.078144, 20 to 84.424280, 30 fc rgb "#0000FF"
set object 36 rect from 0.078682, 20 to 85.308658, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079500, 20 to 86.558394, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080680, 20 to 87.840366, 30 fc rgb "#808080"
set object 39 rect from 0.081873, 20 to 88.521649, 30 fc rgb "#800080"
set object 40 rect from 0.082747, 20 to 89.190035, 30 fc rgb "#008080"
set object 41 rect from 0.083132, 20 to 149.289077, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.137810, 30 to 152.546124, 40 fc rgb "#FF0000"
set object 43 rect from 0.058406, 30 to 64.966881, 40 fc rgb "#00FF00"
set object 44 rect from 0.060894, 30 to 65.967314, 40 fc rgb "#0000FF"
set object 45 rect from 0.061500, 30 to 67.003209, 40 fc rgb "#FFFF00"
set object 46 rect from 0.062466, 30 to 68.347506, 40 fc rgb "#FF00FF"
set object 47 rect from 0.063730, 30 to 69.637001, 40 fc rgb "#808080"
set object 48 rect from 0.064947, 30 to 70.289271, 40 fc rgb "#800080"
set object 49 rect from 0.065789, 30 to 71.044699, 40 fc rgb "#008080"
set object 50 rect from 0.066267, 30 to 147.438652, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.136057, 40 to 154.304137, 50 fc rgb "#FF0000"
set object 52 rect from 0.034527, 40 to 40.610479, 50 fc rgb "#00FF00"
set object 53 rect from 0.038242, 40 to 42.349148, 50 fc rgb "#0000FF"
set object 54 rect from 0.039542, 40 to 45.019477, 50 fc rgb "#FFFF00"
set object 55 rect from 0.042052, 40 to 47.143918, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044005, 40 to 48.817037, 50 fc rgb "#808080"
set object 57 rect from 0.045579, 40 to 49.664881, 50 fc rgb "#800080"
set object 58 rect from 0.046788, 40 to 51.051087, 50 fc rgb "#008080"
set object 59 rect from 0.047643, 40 to 145.110041, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.144356, 50 to 159.251501, 60 fc rgb "#FF0000"
set object 61 rect from 0.122258, 50 to 133.335880, 60 fc rgb "#00FF00"
set object 62 rect from 0.124448, 50 to 134.055849, 60 fc rgb "#0000FF"
set object 63 rect from 0.124865, 50 to 135.047686, 60 fc rgb "#FFFF00"
set object 64 rect from 0.125787, 50 to 136.327508, 60 fc rgb "#FF00FF"
set object 65 rect from 0.126990, 50 to 137.558975, 60 fc rgb "#808080"
set object 66 rect from 0.128142, 50 to 138.234886, 60 fc rgb "#800080"
set object 67 rect from 0.129004, 50 to 138.954853, 60 fc rgb "#008080"
set object 68 rect from 0.129447, 50 to 154.585677, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
