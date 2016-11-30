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

set object 15 rect from 0.139778, 0 to 160.102408, 10 fc rgb "#FF0000"
set object 16 rect from 0.116885, 0 to 132.702303, 10 fc rgb "#00FF00"
set object 17 rect from 0.119275, 0 to 133.475560, 10 fc rgb "#0000FF"
set object 18 rect from 0.119734, 0 to 134.344787, 10 fc rgb "#FFFF00"
set object 19 rect from 0.120530, 0 to 135.558826, 10 fc rgb "#FF00FF"
set object 20 rect from 0.121624, 0 to 137.043985, 10 fc rgb "#808080"
set object 21 rect from 0.122949, 0 to 137.664372, 10 fc rgb "#800080"
set object 22 rect from 0.123771, 0 to 138.326061, 10 fc rgb "#008080"
set object 23 rect from 0.124085, 0 to 155.313293, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.131463, 10 to 153.584880, 20 fc rgb "#FF0000"
set object 25 rect from 0.028802, 10 to 35.504282, 20 fc rgb "#00FF00"
set object 26 rect from 0.032280, 10 to 37.028514, 20 fc rgb "#0000FF"
set object 27 rect from 0.033323, 10 to 38.929848, 20 fc rgb "#FFFF00"
set object 28 rect from 0.035058, 10 to 40.804445, 20 fc rgb "#FF00FF"
set object 29 rect from 0.036704, 10 to 42.557399, 20 fc rgb "#808080"
set object 30 rect from 0.038286, 10 to 43.418812, 20 fc rgb "#800080"
set object 31 rect from 0.039692, 10 to 44.505620, 20 fc rgb "#008080"
set object 32 rect from 0.040044, 10 to 145.155977, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.133391, 20 to 154.024532, 30 fc rgb "#FF0000"
set object 34 rect from 0.049937, 20 to 58.371926, 30 fc rgb "#00FF00"
set object 35 rect from 0.052682, 20 to 59.135140, 30 fc rgb "#0000FF"
set object 36 rect from 0.053105, 20 to 60.281074, 30 fc rgb "#FFFF00"
set object 37 rect from 0.054173, 20 to 62.152313, 30 fc rgb "#FF00FF"
set object 38 rect from 0.055840, 20 to 63.654199, 30 fc rgb "#808080"
set object 39 rect from 0.057194, 20 to 64.447574, 30 fc rgb "#800080"
set object 40 rect from 0.058194, 20 to 65.676078, 30 fc rgb "#008080"
set object 41 rect from 0.058998, 20 to 148.087255, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.138117, 30 to 158.192162, 40 fc rgb "#FF0000"
set object 43 rect from 0.101657, 30 to 115.225207, 40 fc rgb "#00FF00"
set object 44 rect from 0.103618, 30 to 115.975052, 40 fc rgb "#0000FF"
set object 45 rect from 0.104070, 30 to 116.903405, 40 fc rgb "#FFFF00"
set object 46 rect from 0.104892, 30 to 118.095097, 40 fc rgb "#FF00FF"
set object 47 rect from 0.105953, 30 to 119.434072, 40 fc rgb "#808080"
set object 48 rect from 0.107152, 30 to 120.111390, 40 fc rgb "#800080"
set object 49 rect from 0.108060, 30 to 120.846703, 40 fc rgb "#008080"
set object 50 rect from 0.108421, 30 to 153.543612, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.136614, 40 to 156.944670, 50 fc rgb "#FF0000"
set object 52 rect from 0.085516, 40 to 97.349760, 50 fc rgb "#00FF00"
set object 53 rect from 0.087611, 40 to 98.395300, 50 fc rgb "#0000FF"
set object 54 rect from 0.088296, 40 to 99.277896, 50 fc rgb "#FFFF00"
set object 55 rect from 0.089098, 40 to 100.689430, 50 fc rgb "#FF00FF"
set object 56 rect from 0.090352, 40 to 101.983778, 50 fc rgb "#808080"
set object 57 rect from 0.091513, 40 to 102.662193, 50 fc rgb "#800080"
set object 58 rect from 0.092413, 40 to 103.399735, 50 fc rgb "#008080"
set object 59 rect from 0.092786, 40 to 151.815200, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.135049, 50 to 155.481824, 60 fc rgb "#FF0000"
set object 61 rect from 0.068227, 50 to 78.138697, 60 fc rgb "#00FF00"
set object 62 rect from 0.070410, 50 to 79.136219, 60 fc rgb "#0000FF"
set object 63 rect from 0.071036, 50 to 80.209659, 60 fc rgb "#FFFF00"
set object 64 rect from 0.072008, 50 to 81.678058, 60 fc rgb "#FF00FF"
set object 65 rect from 0.073315, 50 to 83.034890, 60 fc rgb "#808080"
set object 66 rect from 0.074533, 50 to 83.754607, 60 fc rgb "#800080"
set object 67 rect from 0.075444, 50 to 84.477650, 60 fc rgb "#008080"
set object 68 rect from 0.075827, 50 to 149.958493, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
