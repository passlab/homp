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

set object 15 rect from 0.117963, 0 to 161.556339, 10 fc rgb "#FF0000"
set object 16 rect from 0.097151, 0 to 132.662654, 10 fc rgb "#00FF00"
set object 17 rect from 0.099434, 0 to 133.553208, 10 fc rgb "#0000FF"
set object 18 rect from 0.099843, 0 to 134.549525, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100602, 0 to 135.938237, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101627, 0 to 136.544871, 10 fc rgb "#808080"
set object 21 rect from 0.102079, 0 to 137.223821, 10 fc rgb "#800080"
set object 22 rect from 0.102848, 0 to 137.953698, 10 fc rgb "#008080"
set object 23 rect from 0.103130, 0 to 157.214837, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.116353, 10 to 159.794068, 20 fc rgb "#FF0000"
set object 25 rect from 0.082459, 10 to 112.979665, 20 fc rgb "#00FF00"
set object 26 rect from 0.084802, 10 to 114.010822, 20 fc rgb "#0000FF"
set object 27 rect from 0.085249, 10 to 115.047328, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086059, 10 to 116.632917, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087237, 10 to 117.314542, 20 fc rgb "#808080"
set object 30 rect from 0.087745, 10 to 118.123400, 20 fc rgb "#800080"
set object 31 rect from 0.088592, 10 to 118.928227, 20 fc rgb "#008080"
set object 32 rect from 0.088928, 10 to 154.999866, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.114668, 20 to 157.209489, 30 fc rgb "#FF0000"
set object 34 rect from 0.069097, 20 to 94.848779, 30 fc rgb "#00FF00"
set object 35 rect from 0.071180, 20 to 95.612100, 30 fc rgb "#0000FF"
set object 36 rect from 0.071527, 20 to 96.777156, 30 fc rgb "#FFFF00"
set object 37 rect from 0.072389, 20 to 98.246208, 30 fc rgb "#FF00FF"
set object 38 rect from 0.073478, 20 to 98.763123, 30 fc rgb "#808080"
set object 39 rect from 0.073866, 20 to 99.464863, 30 fc rgb "#800080"
set object 40 rect from 0.074651, 20 to 100.238880, 30 fc rgb "#008080"
set object 41 rect from 0.074988, 20 to 152.879960, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.113059, 30 to 155.366799, 40 fc rgb "#FF0000"
set object 43 rect from 0.054309, 30 to 75.118896, 40 fc rgb "#00FF00"
set object 44 rect from 0.056455, 30 to 76.079093, 40 fc rgb "#0000FF"
set object 45 rect from 0.056926, 30 to 77.316466, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057856, 30 to 78.778813, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058941, 30 to 79.318517, 40 fc rgb "#808080"
set object 48 rect from 0.059362, 30 to 80.056375, 40 fc rgb "#800080"
set object 49 rect from 0.060164, 30 to 80.838454, 40 fc rgb "#008080"
set object 50 rect from 0.060481, 30 to 150.572597, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.111427, 40 to 153.249568, 50 fc rgb "#FF0000"
set object 52 rect from 0.039983, 40 to 56.347892, 50 fc rgb "#00FF00"
set object 53 rect from 0.042524, 40 to 57.376336, 50 fc rgb "#0000FF"
set object 54 rect from 0.042978, 40 to 58.605687, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043887, 40 to 60.187245, 50 fc rgb "#FF00FF"
set object 56 rect from 0.045071, 40 to 60.780509, 50 fc rgb "#808080"
set object 57 rect from 0.045519, 40 to 61.515693, 50 fc rgb "#800080"
set object 58 rect from 0.046337, 40 to 62.344626, 50 fc rgb "#008080"
set object 59 rect from 0.046690, 40 to 148.435291, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.109663, 50 to 155.202090, 60 fc rgb "#FF0000"
set object 61 rect from 0.019051, 50 to 29.460209, 60 fc rgb "#00FF00"
set object 62 rect from 0.022466, 50 to 31.367154, 60 fc rgb "#0000FF"
set object 63 rect from 0.023555, 50 to 34.875772, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026203, 50 to 37.071986, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027813, 50 to 38.013425, 60 fc rgb "#808080"
set object 66 rect from 0.028543, 50 to 38.993698, 60 fc rgb "#800080"
set object 67 rect from 0.029700, 50 to 40.619436, 60 fc rgb "#008080"
set object 68 rect from 0.030483, 50 to 145.545399, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
