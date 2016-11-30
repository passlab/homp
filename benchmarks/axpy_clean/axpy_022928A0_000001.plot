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

set object 15 rect from 0.124770, 0 to 161.690326, 10 fc rgb "#FF0000"
set object 16 rect from 0.103994, 0 to 132.559235, 10 fc rgb "#00FF00"
set object 17 rect from 0.105828, 0 to 133.358661, 10 fc rgb "#0000FF"
set object 18 rect from 0.106217, 0 to 134.199564, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106929, 0 to 135.514327, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107935, 0 to 137.730332, 10 fc rgb "#808080"
set object 21 rect from 0.109698, 0 to 138.396521, 10 fc rgb "#800080"
set object 22 rect from 0.110487, 0 to 139.082808, 10 fc rgb "#008080"
set object 23 rect from 0.110774, 0 to 155.987541, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.123109, 10 to 160.166892, 20 fc rgb "#FF0000"
set object 25 rect from 0.088855, 10 to 113.694931, 20 fc rgb "#00FF00"
set object 26 rect from 0.090820, 10 to 114.550911, 20 fc rgb "#0000FF"
set object 27 rect from 0.091266, 10 to 115.571557, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092097, 10 to 117.087433, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093302, 10 to 119.361258, 20 fc rgb "#808080"
set object 30 rect from 0.095115, 10 to 120.170741, 20 fc rgb "#800080"
set object 31 rect from 0.095989, 10 to 120.913593, 20 fc rgb "#008080"
set object 32 rect from 0.096320, 10 to 153.879636, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.121327, 20 to 159.310920, 30 fc rgb "#FF0000"
set object 34 rect from 0.073805, 20 to 94.633277, 30 fc rgb "#00FF00"
set object 35 rect from 0.075688, 20 to 95.407557, 30 fc rgb "#0000FF"
set object 36 rect from 0.076024, 20 to 98.279693, 30 fc rgb "#FFFF00"
set object 37 rect from 0.078316, 20 to 99.747808, 30 fc rgb "#FF00FF"
set object 38 rect from 0.079475, 20 to 101.637002, 30 fc rgb "#808080"
set object 39 rect from 0.080992, 20 to 102.421341, 30 fc rgb "#800080"
set object 40 rect from 0.081882, 20 to 103.248412, 30 fc rgb "#008080"
set object 41 rect from 0.082264, 20 to 151.810690, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119761, 30 to 157.244475, 40 fc rgb "#FF0000"
set object 43 rect from 0.057565, 30 to 74.284505, 40 fc rgb "#00FF00"
set object 44 rect from 0.059469, 30 to 75.047473, 40 fc rgb "#0000FF"
set object 45 rect from 0.059825, 30 to 77.599082, 40 fc rgb "#FFFF00"
set object 46 rect from 0.061855, 30 to 79.186610, 40 fc rgb "#FF00FF"
set object 47 rect from 0.063117, 30 to 81.108488, 40 fc rgb "#808080"
set object 48 rect from 0.064663, 30 to 81.906649, 40 fc rgb "#800080"
set object 49 rect from 0.065567, 30 to 82.679675, 40 fc rgb "#008080"
set object 50 rect from 0.065899, 30 to 149.712843, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.118121, 40 to 155.312558, 50 fc rgb "#FF0000"
set object 52 rect from 0.041635, 40 to 54.723846, 50 fc rgb "#00FF00"
set object 53 rect from 0.043902, 40 to 55.508185, 50 fc rgb "#0000FF"
set object 54 rect from 0.044280, 40 to 58.042197, 50 fc rgb "#FFFF00"
set object 55 rect from 0.046329, 40 to 59.757932, 50 fc rgb "#FF00FF"
set object 56 rect from 0.047674, 40 to 61.677291, 50 fc rgb "#808080"
set object 57 rect from 0.049189, 40 to 62.534536, 50 fc rgb "#800080"
set object 58 rect from 0.050182, 40 to 63.418172, 50 fc rgb "#008080"
set object 59 rect from 0.050590, 40 to 147.643897, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116331, 50 to 155.611704, 60 fc rgb "#FF0000"
set object 61 rect from 0.021149, 50 to 29.834974, 60 fc rgb "#00FF00"
set object 62 rect from 0.024210, 50 to 31.119573, 60 fc rgb "#0000FF"
set object 63 rect from 0.024893, 50 to 35.173236, 60 fc rgb "#FFFF00"
set object 64 rect from 0.028149, 50 to 37.170530, 60 fc rgb "#FF00FF"
set object 65 rect from 0.029705, 50 to 39.384025, 60 fc rgb "#808080"
set object 66 rect from 0.031499, 50 to 40.405926, 60 fc rgb "#800080"
set object 67 rect from 0.032706, 50 to 41.997219, 60 fc rgb "#008080"
set object 68 rect from 0.033572, 50 to 145.181536, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
