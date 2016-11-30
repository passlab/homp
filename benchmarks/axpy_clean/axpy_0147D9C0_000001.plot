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

set object 15 rect from 0.144771, 0 to 155.300424, 10 fc rgb "#FF0000"
set object 16 rect from 0.077335, 0 to 82.826409, 10 fc rgb "#00FF00"
set object 17 rect from 0.079858, 0 to 83.797369, 10 fc rgb "#0000FF"
set object 18 rect from 0.080564, 0 to 84.716238, 10 fc rgb "#FFFF00"
set object 19 rect from 0.081473, 0 to 85.978904, 10 fc rgb "#FF00FF"
set object 20 rect from 0.082673, 0 to 87.498892, 10 fc rgb "#808080"
set object 21 rect from 0.084136, 0 to 88.156270, 10 fc rgb "#800080"
set object 22 rect from 0.085002, 0 to 88.765725, 10 fc rgb "#008080"
set object 23 rect from 0.085334, 0 to 150.081001, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.148018, 10 to 158.107043, 20 fc rgb "#FF0000"
set object 25 rect from 0.109213, 10 to 115.612972, 20 fc rgb "#00FF00"
set object 26 rect from 0.111349, 10 to 116.365152, 20 fc rgb "#0000FF"
set object 27 rect from 0.111822, 10 to 117.203800, 20 fc rgb "#FFFF00"
set object 28 rect from 0.112613, 10 to 118.299780, 20 fc rgb "#FF00FF"
set object 29 rect from 0.113692, 10 to 119.682248, 20 fc rgb "#808080"
set object 30 rect from 0.115008, 10 to 120.312541, 20 fc rgb "#800080"
set object 31 rect from 0.115863, 10 to 120.887614, 20 fc rgb "#008080"
set object 32 rect from 0.116164, 10 to 153.525197, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.146350, 20 to 156.649562, 30 fc rgb "#FF0000"
set object 34 rect from 0.094410, 20 to 100.328689, 30 fc rgb "#00FF00"
set object 35 rect from 0.096680, 20 to 101.073576, 30 fc rgb "#0000FF"
set object 36 rect from 0.097131, 20 to 101.967444, 30 fc rgb "#FFFF00"
set object 37 rect from 0.098006, 20 to 103.425968, 30 fc rgb "#FF00FF"
set object 38 rect from 0.099404, 20 to 104.656336, 30 fc rgb "#808080"
set object 39 rect from 0.100573, 20 to 105.292877, 30 fc rgb "#800080"
set object 40 rect from 0.101440, 20 to 105.925250, 30 fc rgb "#008080"
set object 41 rect from 0.101793, 20 to 151.905195, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.149603, 30 to 159.793725, 40 fc rgb "#FF0000"
set object 43 rect from 0.124855, 30 to 131.952597, 40 fc rgb "#00FF00"
set object 44 rect from 0.127018, 30 to 132.580805, 40 fc rgb "#0000FF"
set object 45 rect from 0.127374, 30 to 133.475713, 40 fc rgb "#FFFF00"
set object 46 rect from 0.128230, 30 to 134.715457, 40 fc rgb "#FF00FF"
set object 47 rect from 0.129434, 30 to 135.954161, 40 fc rgb "#808080"
set object 48 rect from 0.130636, 30 to 136.654250, 40 fc rgb "#800080"
set object 49 rect from 0.131543, 30 to 137.308504, 40 fc rgb "#008080"
set object 50 rect from 0.131914, 30 to 155.309805, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.142981, 40 to 153.250161, 50 fc rgb "#FF0000"
set object 52 rect from 0.060596, 40 to 65.369971, 50 fc rgb "#00FF00"
set object 53 rect from 0.063107, 40 to 66.197163, 50 fc rgb "#0000FF"
set object 54 rect from 0.063651, 40 to 67.124365, 50 fc rgb "#FFFF00"
set object 55 rect from 0.064545, 40 to 68.523506, 50 fc rgb "#FF00FF"
set object 56 rect from 0.065898, 40 to 69.773667, 50 fc rgb "#808080"
set object 57 rect from 0.067097, 40 to 70.410212, 50 fc rgb "#800080"
set object 58 rect from 0.068015, 40 to 71.133222, 50 fc rgb "#008080"
set object 59 rect from 0.068434, 40 to 148.039067, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.140827, 50 to 154.302382, 60 fc rgb "#FF0000"
set object 61 rect from 0.036783, 50 to 42.279463, 60 fc rgb "#00FF00"
set object 62 rect from 0.041087, 50 to 43.924470, 60 fc rgb "#0000FF"
set object 63 rect from 0.042300, 50 to 46.276859, 60 fc rgb "#FFFF00"
set object 64 rect from 0.044569, 50 to 48.364632, 60 fc rgb "#FF00FF"
set object 65 rect from 0.046548, 50 to 49.987758, 60 fc rgb "#808080"
set object 66 rect from 0.048110, 50 to 50.814950, 60 fc rgb "#800080"
set object 67 rect from 0.049462, 50 to 52.250553, 60 fc rgb "#008080"
set object 68 rect from 0.050293, 50 to 145.183489, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
