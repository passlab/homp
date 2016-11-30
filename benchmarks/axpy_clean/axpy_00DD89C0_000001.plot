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

set object 15 rect from 0.125097, 0 to 161.071916, 10 fc rgb "#FF0000"
set object 16 rect from 0.103940, 0 to 132.630798, 10 fc rgb "#00FF00"
set object 17 rect from 0.106203, 0 to 133.385159, 10 fc rgb "#0000FF"
set object 18 rect from 0.106562, 0 to 134.273588, 10 fc rgb "#FFFF00"
set object 19 rect from 0.107274, 0 to 135.556750, 10 fc rgb "#FF00FF"
set object 20 rect from 0.108302, 0 to 137.179487, 10 fc rgb "#808080"
set object 21 rect from 0.109609, 0 to 137.863641, 10 fc rgb "#800080"
set object 22 rect from 0.110388, 0 to 138.508993, 10 fc rgb "#008080"
set object 23 rect from 0.110655, 0 to 156.065867, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.123520, 10 to 159.698567, 20 fc rgb "#FF0000"
set object 25 rect from 0.088502, 10 to 113.264465, 20 fc rgb "#00FF00"
set object 26 rect from 0.090756, 10 to 114.211823, 20 fc rgb "#0000FF"
set object 27 rect from 0.091261, 10 to 115.254374, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092126, 10 to 116.771836, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093334, 10 to 118.444689, 20 fc rgb "#808080"
set object 30 rect from 0.094672, 10 to 119.200321, 20 fc rgb "#800080"
set object 31 rect from 0.095514, 10 to 119.917076, 20 fc rgb "#008080"
set object 32 rect from 0.095835, 10 to 153.955670, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121765, 20 to 157.366543, 30 fc rgb "#FF0000"
set object 34 rect from 0.074175, 20 to 95.185065, 30 fc rgb "#00FF00"
set object 35 rect from 0.076345, 20 to 96.005826, 30 fc rgb "#0000FF"
set object 36 rect from 0.076731, 20 to 97.083480, 30 fc rgb "#FFFF00"
set object 37 rect from 0.077591, 20 to 98.633544, 30 fc rgb "#FF00FF"
set object 38 rect from 0.078834, 20 to 100.157280, 30 fc rgb "#808080"
set object 39 rect from 0.080049, 20 to 100.900364, 30 fc rgb "#800080"
set object 40 rect from 0.080895, 20 to 101.681016, 30 fc rgb "#008080"
set object 41 rect from 0.081263, 20 to 151.953288, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.120207, 30 to 155.317816, 40 fc rgb "#FF0000"
set object 43 rect from 0.058814, 30 to 76.075626, 40 fc rgb "#00FF00"
set object 44 rect from 0.061093, 30 to 76.872598, 40 fc rgb "#0000FF"
set object 45 rect from 0.061462, 30 to 78.021655, 40 fc rgb "#FFFF00"
set object 46 rect from 0.062380, 30 to 79.427606, 40 fc rgb "#FF00FF"
set object 47 rect from 0.063499, 30 to 80.925052, 40 fc rgb "#808080"
set object 48 rect from 0.064696, 30 to 81.671870, 40 fc rgb "#800080"
set object 49 rect from 0.065552, 30 to 82.498904, 40 fc rgb "#008080"
set object 50 rect from 0.065959, 30 to 149.955873, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.118625, 40 to 153.514555, 50 fc rgb "#FF0000"
set object 52 rect from 0.043550, 40 to 57.377202, 50 fc rgb "#00FF00"
set object 53 rect from 0.046165, 40 to 58.241805, 50 fc rgb "#0000FF"
set object 54 rect from 0.046592, 40 to 59.358298, 50 fc rgb "#FFFF00"
set object 55 rect from 0.047497, 40 to 60.890810, 50 fc rgb "#FF00FF"
set object 56 rect from 0.048729, 40 to 62.415815, 50 fc rgb "#808080"
set object 57 rect from 0.049934, 40 to 63.196505, 50 fc rgb "#800080"
set object 58 rect from 0.050832, 40 to 64.067382, 50 fc rgb "#008080"
set object 59 rect from 0.051361, 40 to 147.920851, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116894, 50 to 155.410468, 60 fc rgb "#FF0000"
set object 61 rect from 0.022256, 50 to 31.463532, 60 fc rgb "#00FF00"
set object 62 rect from 0.025586, 50 to 33.844402, 60 fc rgb "#0000FF"
set object 63 rect from 0.027161, 50 to 36.249023, 60 fc rgb "#FFFF00"
set object 64 rect from 0.029109, 50 to 38.584781, 60 fc rgb "#FF00FF"
set object 65 rect from 0.030933, 50 to 40.585931, 60 fc rgb "#808080"
set object 66 rect from 0.032535, 50 to 41.607195, 60 fc rgb "#800080"
set object 67 rect from 0.033834, 50 to 43.321426, 60 fc rgb "#008080"
set object 68 rect from 0.034712, 50 to 145.343287, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
