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

set object 15 rect from 0.150239, 0 to 159.901761, 10 fc rgb "#FF0000"
set object 16 rect from 0.094583, 0 to 100.888440, 10 fc rgb "#00FF00"
set object 17 rect from 0.096780, 0 to 101.516076, 10 fc rgb "#0000FF"
set object 18 rect from 0.097154, 0 to 102.299591, 10 fc rgb "#FFFF00"
set object 19 rect from 0.097907, 0 to 103.341493, 10 fc rgb "#FF00FF"
set object 20 rect from 0.098929, 0 to 103.817489, 10 fc rgb "#808080"
set object 21 rect from 0.099362, 0 to 104.369803, 10 fc rgb "#800080"
set object 22 rect from 0.100139, 0 to 104.900169, 10 fc rgb "#008080"
set object 23 rect from 0.100396, 0 to 156.494535, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.148393, 10 to 158.304302, 20 fc rgb "#FF0000"
set object 25 rect from 0.080503, 10 to 86.157338, 20 fc rgb "#00FF00"
set object 26 rect from 0.082787, 10 to 86.939794, 20 fc rgb "#0000FF"
set object 27 rect from 0.083218, 10 to 87.746317, 20 fc rgb "#FFFF00"
set object 28 rect from 0.084034, 10 to 88.973405, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085197, 10 to 89.473468, 20 fc rgb "#808080"
set object 30 rect from 0.085677, 10 to 90.135647, 20 fc rgb "#800080"
set object 31 rect from 0.086556, 10 to 90.749690, 20 fc rgb "#008080"
set object 32 rect from 0.086869, 10 to 154.536245, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.146502, 20 to 156.044790, 30 fc rgb "#FF0000"
set object 34 rect from 0.067545, 20 to 72.434656, 30 fc rgb "#00FF00"
set object 35 rect from 0.069628, 20 to 73.057116, 30 fc rgb "#0000FF"
set object 36 rect from 0.069976, 20 to 73.989154, 30 fc rgb "#FFFF00"
set object 37 rect from 0.070843, 20 to 75.076075, 30 fc rgb "#FF00FF"
set object 38 rect from 0.071879, 20 to 75.482984, 30 fc rgb "#808080"
set object 39 rect from 0.072272, 20 to 76.012290, 30 fc rgb "#800080"
set object 40 rect from 0.073029, 20 to 76.624275, 30 fc rgb "#008080"
set object 41 rect from 0.073365, 20 to 152.734895, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.144653, 30 to 154.224547, 40 fc rgb "#FF0000"
set object 43 rect from 0.053762, 30 to 58.013257, 40 fc rgb "#00FF00"
set object 44 rect from 0.055891, 30 to 58.706798, 40 fc rgb "#0000FF"
set object 45 rect from 0.056240, 30 to 59.653489, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057141, 30 to 60.736232, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058172, 30 to 61.187099, 40 fc rgb "#808080"
set object 48 rect from 0.058606, 30 to 61.739414, 40 fc rgb "#800080"
set object 49 rect from 0.059421, 30 to 62.367049, 40 fc rgb "#008080"
set object 50 rect from 0.059754, 30 to 150.685571, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.142672, 40 to 152.361469, 50 fc rgb "#FF0000"
set object 52 rect from 0.039859, 40 to 43.921326, 50 fc rgb "#00FF00"
set object 53 rect from 0.042375, 40 to 44.570909, 50 fc rgb "#0000FF"
set object 54 rect from 0.042727, 40 to 45.555323, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043662, 40 to 46.760400, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044838, 40 to 47.251048, 50 fc rgb "#808080"
set object 57 rect from 0.045298, 40 to 47.821133, 50 fc rgb "#800080"
set object 58 rect from 0.046153, 40 to 48.531447, 50 fc rgb "#008080"
set object 59 rect from 0.046525, 40 to 148.465777, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.140394, 50 to 152.870822, 60 fc rgb "#FF0000"
set object 61 rect from 0.020464, 50 to 24.351021, 60 fc rgb "#00FF00"
set object 62 rect from 0.023794, 50 to 25.640835, 60 fc rgb "#0000FF"
set object 63 rect from 0.024651, 50 to 28.142024, 60 fc rgb "#FFFF00"
set object 64 rect from 0.027122, 50 to 29.945494, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028749, 50 to 30.651631, 60 fc rgb "#808080"
set object 66 rect from 0.029436, 50 to 31.397485, 60 fc rgb "#800080"
set object 67 rect from 0.030576, 50 to 32.504234, 60 fc rgb "#008080"
set object 68 rect from 0.031219, 50 to 145.868316, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
