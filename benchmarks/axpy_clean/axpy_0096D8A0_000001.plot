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

set object 15 rect from 0.112270, 0 to 162.604729, 10 fc rgb "#FF0000"
set object 16 rect from 0.093095, 0 to 132.521903, 10 fc rgb "#00FF00"
set object 17 rect from 0.094839, 0 to 133.315611, 10 fc rgb "#0000FF"
set object 18 rect from 0.095178, 0 to 134.165425, 10 fc rgb "#FFFF00"
set object 19 rect from 0.095785, 0 to 135.515876, 10 fc rgb "#FF00FF"
set object 20 rect from 0.096749, 0 to 137.947527, 10 fc rgb "#808080"
set object 21 rect from 0.098499, 0 to 138.655705, 10 fc rgb "#800080"
set object 22 rect from 0.099251, 0 to 139.379306, 10 fc rgb "#008080"
set object 23 rect from 0.099514, 0 to 156.698133, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.110846, 10 to 161.118264, 20 fc rgb "#FF0000"
set object 25 rect from 0.079162, 10 to 113.088361, 20 fc rgb "#00FF00"
set object 26 rect from 0.080985, 10 to 113.952197, 20 fc rgb "#0000FF"
set object 27 rect from 0.081373, 10 to 114.954869, 20 fc rgb "#FFFF00"
set object 28 rect from 0.082132, 10 to 116.578769, 20 fc rgb "#FF00FF"
set object 29 rect from 0.083276, 10 to 119.030042, 20 fc rgb "#808080"
set object 30 rect from 0.085026, 10 to 119.872856, 20 fc rgb "#800080"
set object 31 rect from 0.085870, 10 to 120.689006, 20 fc rgb "#008080"
set object 32 rect from 0.086200, 10 to 154.594618, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.109220, 20 to 160.233406, 30 fc rgb "#FF0000"
set object 34 rect from 0.065504, 20 to 93.691283, 30 fc rgb "#00FF00"
set object 35 rect from 0.067168, 20 to 94.515854, 30 fc rgb "#0000FF"
set object 36 rect from 0.067511, 20 to 97.354191, 30 fc rgb "#FFFF00"
set object 37 rect from 0.069566, 20 to 98.976691, 30 fc rgb "#FF00FF"
set object 38 rect from 0.070693, 20 to 100.962392, 30 fc rgb "#808080"
set object 39 rect from 0.072106, 20 to 101.757500, 30 fc rgb "#800080"
set object 40 rect from 0.072920, 20 to 102.535807, 30 fc rgb "#008080"
set object 41 rect from 0.073239, 20 to 152.576674, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.107877, 30 to 158.186019, 40 fc rgb "#FF0000"
set object 43 rect from 0.050264, 30 to 72.500625, 40 fc rgb "#00FF00"
set object 44 rect from 0.052055, 30 to 73.305575, 40 fc rgb "#0000FF"
set object 45 rect from 0.052384, 30 to 75.905505, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054281, 30 to 77.606534, 40 fc rgb "#FF00FF"
set object 47 rect from 0.055452, 30 to 79.613256, 40 fc rgb "#808080"
set object 48 rect from 0.056898, 30 to 80.454671, 40 fc rgb "#800080"
set object 49 rect from 0.057737, 30 to 81.319907, 40 fc rgb "#008080"
set object 50 rect from 0.058112, 30 to 150.547488, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.106356, 40 to 156.224119, 50 fc rgb "#FF0000"
set object 52 rect from 0.034809, 40 to 51.333811, 50 fc rgb "#00FF00"
set object 53 rect from 0.036951, 40 to 52.155582, 50 fc rgb "#0000FF"
set object 54 rect from 0.037304, 40 to 54.791976, 50 fc rgb "#FFFF00"
set object 55 rect from 0.039222, 40 to 56.532269, 50 fc rgb "#FF00FF"
set object 56 rect from 0.040434, 40 to 58.538992, 50 fc rgb "#808080"
set object 57 rect from 0.041867, 40 to 59.450514, 50 fc rgb "#800080"
set object 58 rect from 0.042780, 40 to 60.384478, 50 fc rgb "#008080"
set object 59 rect from 0.043192, 40 to 148.260294, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.104622, 50 to 156.292826, 60 fc rgb "#FF0000"
set object 61 rect from 0.015198, 50 to 24.636201, 60 fc rgb "#00FF00"
set object 62 rect from 0.018060, 50 to 26.119866, 60 fc rgb "#0000FF"
set object 63 rect from 0.018750, 50 to 30.007118, 60 fc rgb "#FFFF00"
set object 64 rect from 0.021548, 50 to 32.235426, 60 fc rgb "#FF00FF"
set object 65 rect from 0.023109, 50 to 34.453933, 60 fc rgb "#808080"
set object 66 rect from 0.024709, 50 to 35.491648, 60 fc rgb "#800080"
set object 67 rect from 0.025793, 50 to 37.053863, 60 fc rgb "#008080"
set object 68 rect from 0.026552, 50 to 145.280322, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
