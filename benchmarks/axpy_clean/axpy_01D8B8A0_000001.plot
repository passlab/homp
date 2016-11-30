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

set object 15 rect from 0.122715, 0 to 161.908591, 10 fc rgb "#FF0000"
set object 16 rect from 0.102127, 0 to 132.629430, 10 fc rgb "#00FF00"
set object 17 rect from 0.104107, 0 to 133.379885, 10 fc rgb "#0000FF"
set object 18 rect from 0.104448, 0 to 134.313148, 10 fc rgb "#FFFF00"
set object 19 rect from 0.105180, 0 to 135.649137, 10 fc rgb "#FF00FF"
set object 20 rect from 0.106225, 0 to 137.908159, 10 fc rgb "#808080"
set object 21 rect from 0.107994, 0 to 138.584470, 10 fc rgb "#800080"
set object 22 rect from 0.108795, 0 to 139.286346, 10 fc rgb "#008080"
set object 23 rect from 0.109070, 0 to 156.236089, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.121218, 10 to 160.562391, 20 fc rgb "#FF0000"
set object 25 rect from 0.087123, 10 to 113.512697, 20 fc rgb "#00FF00"
set object 26 rect from 0.089169, 10 to 114.398658, 20 fc rgb "#0000FF"
set object 27 rect from 0.089601, 10 to 115.500689, 20 fc rgb "#FFFF00"
set object 28 rect from 0.090509, 10 to 117.056577, 20 fc rgb "#FF00FF"
set object 29 rect from 0.091710, 10 to 119.398698, 20 fc rgb "#808080"
set object 30 rect from 0.093525, 10 to 120.158106, 20 fc rgb "#800080"
set object 31 rect from 0.094417, 10 to 120.982705, 20 fc rgb "#008080"
set object 32 rect from 0.094755, 10 to 154.213578, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.119568, 20 to 159.470590, 30 fc rgb "#FF0000"
set object 34 rect from 0.072116, 20 to 94.320544, 30 fc rgb "#00FF00"
set object 35 rect from 0.074173, 20 to 95.102946, 30 fc rgb "#0000FF"
set object 36 rect from 0.074507, 20 to 97.575485, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076483, 20 to 99.141585, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077670, 20 to 101.170497, 30 fc rgb "#808080"
set object 39 rect from 0.079269, 20 to 101.969530, 30 fc rgb "#800080"
set object 40 rect from 0.080151, 20 to 102.769840, 30 fc rgb "#008080"
set object 41 rect from 0.080510, 20 to 152.292054, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.118135, 30 to 157.758754, 40 fc rgb "#FF0000"
set object 43 rect from 0.054609, 30 to 71.762212, 40 fc rgb "#00FF00"
set object 44 rect from 0.056496, 30 to 72.559969, 40 fc rgb "#0000FF"
set object 45 rect from 0.056905, 30 to 75.203811, 40 fc rgb "#FFFF00"
set object 46 rect from 0.058958, 30 to 76.759680, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060178, 30 to 78.714448, 40 fc rgb "#808080"
set object 48 rect from 0.061721, 30 to 79.536494, 40 fc rgb "#800080"
set object 49 rect from 0.062639, 30 to 80.384106, 40 fc rgb "#008080"
set object 50 rect from 0.063006, 30 to 150.278497, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.116589, 40 to 155.806520, 50 fc rgb "#FF0000"
set object 52 rect from 0.039222, 40 to 52.522737, 50 fc rgb "#00FF00"
set object 53 rect from 0.041448, 40 to 53.357566, 50 fc rgb "#0000FF"
set object 54 rect from 0.041857, 40 to 55.766192, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043772, 40 to 57.492087, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045122, 40 to 59.460913, 50 fc rgb "#808080"
set object 57 rect from 0.046629, 40 to 60.288084, 50 fc rgb "#800080"
set object 58 rect from 0.047561, 40 to 61.153585, 50 fc rgb "#008080"
set object 59 rect from 0.048010, 40 to 148.326282, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.114578, 50 to 155.763028, 60 fc rgb "#FF0000"
set object 61 rect from 0.017899, 50 to 26.212152, 60 fc rgb "#00FF00"
set object 62 rect from 0.021034, 50 to 27.597996, 60 fc rgb "#0000FF"
set object 63 rect from 0.021727, 50 to 31.396276, 60 fc rgb "#FFFF00"
set object 64 rect from 0.024734, 50 to 33.578601, 60 fc rgb "#FF00FF"
set object 65 rect from 0.026411, 50 to 35.766032, 60 fc rgb "#808080"
set object 66 rect from 0.028136, 50 to 36.792642, 60 fc rgb "#800080"
set object 67 rect from 0.029361, 50 to 38.386879, 60 fc rgb "#008080"
set object 68 rect from 0.030168, 50 to 145.307852, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
