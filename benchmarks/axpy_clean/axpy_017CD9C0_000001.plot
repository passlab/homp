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

set object 15 rect from 0.112838, 0 to 162.925914, 10 fc rgb "#FF0000"
set object 16 rect from 0.092908, 0 to 132.593614, 10 fc rgb "#00FF00"
set object 17 rect from 0.094788, 0 to 133.462831, 10 fc rgb "#0000FF"
set object 18 rect from 0.095164, 0 to 134.358705, 10 fc rgb "#FFFF00"
set object 19 rect from 0.095803, 0 to 135.799428, 10 fc rgb "#FF00FF"
set object 20 rect from 0.096829, 0 to 137.346864, 10 fc rgb "#808080"
set object 21 rect from 0.097935, 0 to 138.092502, 10 fc rgb "#800080"
set object 22 rect from 0.098735, 0 to 138.854984, 10 fc rgb "#008080"
set object 23 rect from 0.099008, 0 to 157.737354, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.111351, 10 to 161.344747, 20 fc rgb "#FF0000"
set object 25 rect from 0.078587, 10 to 112.551388, 20 fc rgb "#00FF00"
set object 26 rect from 0.080534, 10 to 113.472539, 20 fc rgb "#0000FF"
set object 27 rect from 0.080932, 10 to 114.475148, 20 fc rgb "#FFFF00"
set object 28 rect from 0.081687, 10 to 116.234611, 20 fc rgb "#FF00FF"
set object 29 rect from 0.082923, 10 to 117.825570, 20 fc rgb "#808080"
set object 30 rect from 0.084056, 10 to 118.715837, 20 fc rgb "#800080"
set object 31 rect from 0.084938, 10 to 119.573817, 20 fc rgb "#008080"
set object 32 rect from 0.085276, 10 to 155.411973, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.109633, 20 to 158.379082, 30 fc rgb "#FF0000"
set object 34 rect from 0.066021, 20 to 94.626682, 30 fc rgb "#00FF00"
set object 35 rect from 0.067760, 20 to 95.467797, 30 fc rgb "#0000FF"
set object 36 rect from 0.068106, 20 to 96.505517, 30 fc rgb "#FFFF00"
set object 37 rect from 0.068845, 20 to 98.079632, 30 fc rgb "#FF00FF"
set object 38 rect from 0.069967, 20 to 99.329379, 30 fc rgb "#808080"
set object 39 rect from 0.070858, 20 to 100.084830, 30 fc rgb "#800080"
set object 40 rect from 0.071651, 20 to 100.907700, 30 fc rgb "#008080"
set object 41 rect from 0.072002, 20 to 153.208789, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.108116, 30 to 156.778288, 40 fc rgb "#FF0000"
set object 43 rect from 0.051923, 30 to 75.074503, 40 fc rgb "#00FF00"
set object 44 rect from 0.053840, 30 to 75.925453, 40 fc rgb "#0000FF"
set object 45 rect from 0.054190, 30 to 77.232762, 40 fc rgb "#FFFF00"
set object 46 rect from 0.055121, 30 to 78.944496, 40 fc rgb "#FF00FF"
set object 47 rect from 0.056341, 30 to 80.299555, 40 fc rgb "#808080"
set object 48 rect from 0.057307, 30 to 81.074675, 40 fc rgb "#800080"
set object 49 rect from 0.058140, 30 to 81.995847, 40 fc rgb "#008080"
set object 50 rect from 0.058530, 30 to 150.971876, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.106559, 40 to 154.611597, 50 fc rgb "#FF0000"
set object 52 rect from 0.037617, 40 to 55.412807, 50 fc rgb "#00FF00"
set object 53 rect from 0.039831, 40 to 56.293239, 50 fc rgb "#0000FF"
set object 54 rect from 0.040210, 40 to 57.448910, 50 fc rgb "#FFFF00"
set object 55 rect from 0.041035, 40 to 59.067949, 50 fc rgb "#FF00FF"
set object 56 rect from 0.042206, 40 to 60.583100, 50 fc rgb "#808080"
set object 57 rect from 0.043276, 40 to 61.473366, 50 fc rgb "#800080"
set object 58 rect from 0.044174, 40 to 62.450678, 50 fc rgb "#008080"
set object 59 rect from 0.044609, 40 to 148.772898, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.104819, 50 to 155.014578, 60 fc rgb "#FF0000"
set object 61 rect from 0.017774, 50 to 28.440805, 60 fc rgb "#00FF00"
set object 62 rect from 0.020923, 50 to 30.347721, 60 fc rgb "#0000FF"
set object 63 rect from 0.021744, 50 to 32.925826, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023605, 50 to 35.151503, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025171, 50 to 36.802850, 60 fc rgb "#808080"
set object 66 rect from 0.026362, 50 to 37.829334, 60 fc rgb "#800080"
set object 67 rect from 0.027503, 50 to 39.487690, 60 fc rgb "#008080"
set object 68 rect from 0.028264, 50 to 145.357862, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
