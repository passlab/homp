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

set object 15 rect from 0.142971, 0 to 157.702202, 10 fc rgb "#FF0000"
set object 16 rect from 0.104457, 0 to 114.355513, 10 fc rgb "#00FF00"
set object 17 rect from 0.106875, 0 to 115.165629, 10 fc rgb "#0000FF"
set object 18 rect from 0.107338, 0 to 116.016571, 10 fc rgb "#FFFF00"
set object 19 rect from 0.108109, 0 to 117.248941, 10 fc rgb "#FF00FF"
set object 20 rect from 0.109251, 0 to 118.763858, 10 fc rgb "#808080"
set object 21 rect from 0.110668, 0 to 119.373078, 10 fc rgb "#800080"
set object 22 rect from 0.111491, 0 to 120.006986, 10 fc rgb "#008080"
set object 23 rect from 0.111814, 0 to 152.985437, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.136049, 10 to 153.023029, 20 fc rgb "#FF0000"
set object 25 rect from 0.032819, 10 to 38.737384, 20 fc rgb "#00FF00"
set object 26 rect from 0.036582, 10 to 40.327517, 20 fc rgb "#0000FF"
set object 27 rect from 0.037684, 10 to 42.210989, 20 fc rgb "#FFFF00"
set object 28 rect from 0.039460, 10 to 43.947263, 20 fc rgb "#FF00FF"
set object 29 rect from 0.041044, 10 to 45.908097, 20 fc rgb "#808080"
set object 30 rect from 0.042896, 10 to 46.685968, 20 fc rgb "#800080"
set object 31 rect from 0.044062, 10 to 47.607835, 20 fc rgb "#008080"
set object 32 rect from 0.044462, 10 to 145.144249, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.138046, 20 to 153.488285, 30 fc rgb "#FF0000"
set object 34 rect from 0.054146, 20 to 60.645998, 30 fc rgb "#00FF00"
set object 35 rect from 0.056958, 20 to 61.609748, 30 fc rgb "#0000FF"
set object 36 rect from 0.057481, 20 to 62.829245, 30 fc rgb "#FFFF00"
set object 37 rect from 0.058626, 20 to 64.671859, 30 fc rgb "#FF00FF"
set object 38 rect from 0.060351, 20 to 66.019214, 30 fc rgb "#808080"
set object 39 rect from 0.061624, 20 to 66.898078, 30 fc rgb "#800080"
set object 40 rect from 0.062655, 20 to 68.050940, 30 fc rgb "#008080"
set object 41 rect from 0.063496, 20 to 147.581066, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.139730, 30 to 154.314507, 40 fc rgb "#FF0000"
set object 43 rect from 0.071667, 30 to 79.131504, 40 fc rgb "#00FF00"
set object 44 rect from 0.074056, 30 to 80.002875, 40 fc rgb "#0000FF"
set object 45 rect from 0.074575, 30 to 80.942994, 40 fc rgb "#FFFF00"
set object 46 rect from 0.075450, 30 to 82.116253, 40 fc rgb "#FF00FF"
set object 47 rect from 0.076542, 30 to 83.447469, 40 fc rgb "#808080"
set object 48 rect from 0.077817, 30 to 84.149068, 40 fc rgb "#800080"
set object 49 rect from 0.078712, 30 to 84.916213, 40 fc rgb "#008080"
set object 50 rect from 0.079151, 30 to 149.457013, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.144531, 40 to 159.382536, 50 fc rgb "#FF0000"
set object 52 rect from 0.121690, 40 to 132.785143, 50 fc rgb "#00FF00"
set object 53 rect from 0.123960, 40 to 133.475983, 50 fc rgb "#0000FF"
set object 54 rect from 0.124345, 40 to 134.440822, 50 fc rgb "#FFFF00"
set object 55 rect from 0.125257, 40 to 135.731212, 50 fc rgb "#FF00FF"
set object 56 rect from 0.126458, 40 to 137.022659, 50 fc rgb "#808080"
set object 57 rect from 0.127659, 40 to 137.696336, 50 fc rgb "#800080"
set object 58 rect from 0.128525, 40 to 138.391499, 50 fc rgb "#008080"
set object 59 rect from 0.128925, 40 to 154.762537, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.141294, 50 to 155.973324, 60 fc rgb "#FF0000"
set object 61 rect from 0.089023, 50 to 97.666448, 60 fc rgb "#00FF00"
set object 62 rect from 0.091269, 50 to 98.325044, 60 fc rgb "#0000FF"
set object 63 rect from 0.091630, 50 to 99.330709, 60 fc rgb "#FFFF00"
set object 64 rect from 0.092564, 50 to 100.675918, 60 fc rgb "#FF00FF"
set object 65 rect from 0.093816, 50 to 101.958751, 60 fc rgb "#808080"
set object 66 rect from 0.095027, 50 to 102.610943, 60 fc rgb "#800080"
set object 67 rect from 0.095875, 50 to 103.307162, 60 fc rgb "#008080"
set object 68 rect from 0.096268, 50 to 151.220152, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
