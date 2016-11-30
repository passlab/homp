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

set object 15 rect from 0.131785, 0 to 160.589018, 10 fc rgb "#FF0000"
set object 16 rect from 0.110464, 0 to 133.283611, 10 fc rgb "#00FF00"
set object 17 rect from 0.112626, 0 to 134.068924, 10 fc rgb "#0000FF"
set object 18 rect from 0.112957, 0 to 134.957601, 10 fc rgb "#FFFF00"
set object 19 rect from 0.113709, 0 to 136.151610, 10 fc rgb "#FF00FF"
set object 20 rect from 0.114722, 0 to 137.637886, 10 fc rgb "#808080"
set object 21 rect from 0.115974, 0 to 138.252118, 10 fc rgb "#800080"
set object 22 rect from 0.116748, 0 to 138.880607, 10 fc rgb "#008080"
set object 23 rect from 0.117036, 0 to 155.881886, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.130116, 10 to 159.261945, 20 fc rgb "#FF0000"
set object 25 rect from 0.094925, 10 to 114.949317, 20 fc rgb "#00FF00"
set object 26 rect from 0.097105, 10 to 115.719187, 20 fc rgb "#0000FF"
set object 27 rect from 0.097517, 10 to 116.761124, 20 fc rgb "#FFFF00"
set object 28 rect from 0.098426, 10 to 118.197500, 20 fc rgb "#FF00FF"
set object 29 rect from 0.099636, 10 to 119.822780, 20 fc rgb "#808080"
set object 30 rect from 0.101006, 10 to 120.577204, 20 fc rgb "#800080"
set object 31 rect from 0.101893, 10 to 121.291234, 20 fc rgb "#008080"
set object 32 rect from 0.102214, 10 to 153.773062, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.128317, 20 to 156.910754, 30 fc rgb "#FF0000"
set object 34 rect from 0.081405, 20 to 98.616922, 30 fc rgb "#00FF00"
set object 35 rect from 0.083368, 20 to 99.494905, 30 fc rgb "#0000FF"
set object 36 rect from 0.083873, 20 to 100.473874, 30 fc rgb "#FFFF00"
set object 37 rect from 0.084688, 20 to 101.822334, 30 fc rgb "#FF00FF"
set object 38 rect from 0.085820, 20 to 103.218316, 30 fc rgb "#808080"
set object 39 rect from 0.086999, 20 to 103.944226, 30 fc rgb "#800080"
set object 40 rect from 0.087869, 20 to 104.712908, 30 fc rgb "#008080"
set object 41 rect from 0.088270, 20 to 151.854329, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.126776, 30 to 158.467126, 40 fc rgb "#FF0000"
set object 43 rect from 0.059575, 30 to 72.807294, 40 fc rgb "#00FF00"
set object 44 rect from 0.061643, 30 to 73.522511, 40 fc rgb "#0000FF"
set object 45 rect from 0.061998, 30 to 75.613515, 40 fc rgb "#FFFF00"
set object 46 rect from 0.063759, 30 to 79.335737, 40 fc rgb "#FF00FF"
set object 47 rect from 0.066895, 30 to 80.769737, 40 fc rgb "#808080"
set object 48 rect from 0.068176, 30 to 81.593069, 40 fc rgb "#800080"
set object 49 rect from 0.069062, 30 to 82.355811, 40 fc rgb "#008080"
set object 50 rect from 0.069439, 30 to 149.891638, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.125120, 40 to 153.473669, 50 fc rgb "#FF0000"
set object 52 rect from 0.044911, 40 to 55.817895, 50 fc rgb "#00FF00"
set object 53 rect from 0.047351, 40 to 56.578260, 50 fc rgb "#0000FF"
set object 54 rect from 0.047754, 40 to 57.813853, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048781, 40 to 59.388045, 50 fc rgb "#FF00FF"
set object 56 rect from 0.050121, 40 to 60.858875, 50 fc rgb "#808080"
set object 57 rect from 0.051378, 40 to 61.563401, 50 fc rgb "#800080"
set object 58 rect from 0.052233, 40 to 62.390297, 50 fc rgb "#008080"
set object 59 rect from 0.052660, 40 to 147.880236, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.123321, 50 to 160.907421, 60 fc rgb "#FF0000"
set object 61 rect from 0.018350, 50 to 25.195396, 60 fc rgb "#00FF00"
set object 62 rect from 0.021814, 50 to 28.772675, 60 fc rgb "#0000FF"
set object 63 rect from 0.024366, 50 to 36.223061, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030675, 50 to 38.160801, 60 fc rgb "#FF00FF"
set object 65 rect from 0.032253, 50 to 39.921522, 60 fc rgb "#808080"
set object 66 rect from 0.033753, 50 to 40.877918, 60 fc rgb "#800080"
set object 67 rect from 0.034980, 50 to 42.357065, 60 fc rgb "#008080"
set object 68 rect from 0.035945, 50 to 145.294995, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
