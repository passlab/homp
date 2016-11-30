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

set object 15 rect from 0.922192, 0 to 156.026574, 10 fc rgb "#FF0000"
set object 16 rect from 0.395461, 0 to 66.556974, 10 fc rgb "#00FF00"
set object 17 rect from 0.411073, 0 to 67.249362, 10 fc rgb "#0000FF"
set object 18 rect from 0.413737, 0 to 67.974318, 10 fc rgb "#FFFF00"
set object 19 rect from 0.418453, 0 to 69.240710, 10 fc rgb "#FF00FF"
set object 20 rect from 0.426037, 0 to 72.628719, 10 fc rgb "#808080"
set object 21 rect from 0.446961, 0 to 73.246852, 10 fc rgb "#800080"
set object 22 rect from 0.452513, 0 to 73.901950, 10 fc rgb "#008080"
set object 23 rect from 0.454601, 0 to 149.414530, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.945592, 10 to 159.529547, 20 fc rgb "#FF0000"
set object 25 rect from 0.609039, 10 to 100.821039, 20 fc rgb "#00FF00"
set object 26 rect from 0.621415, 10 to 101.430055, 20 fc rgb "#0000FF"
set object 27 rect from 0.623648, 10 to 102.178946, 20 fc rgb "#FFFF00"
set object 28 rect from 0.628308, 10 to 103.264425, 20 fc rgb "#FF00FF"
set object 29 rect from 0.634962, 10 to 106.521839, 20 fc rgb "#808080"
set object 30 rect from 0.654966, 10 to 107.139647, 20 fc rgb "#800080"
set object 31 rect from 0.660345, 10 to 107.697365, 20 fc rgb "#008080"
set object 32 rect from 0.662160, 10 to 153.318248, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.910495, 20 to 155.708713, 30 fc rgb "#FF0000"
set object 34 rect from 0.279411, 20 to 47.415847, 30 fc rgb "#00FF00"
set object 35 rect from 0.293580, 20 to 48.170600, 30 fc rgb "#0000FF"
set object 36 rect from 0.296582, 20 to 50.203798, 30 fc rgb "#FFFF00"
set object 37 rect from 0.309183, 20 to 51.773881, 30 fc rgb "#FF00FF"
set object 38 rect from 0.318769, 20 to 55.050023, 30 fc rgb "#808080"
set object 39 rect from 0.338836, 20 to 55.653499, 30 fc rgb "#800080"
set object 40 rect from 0.344403, 20 to 56.454336, 30 fc rgb "#008080"
set object 41 rect from 0.347585, 20 to 147.583091, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.897649, 30 to 155.580884, 40 fc rgb "#FF0000"
set object 43 rect from 0.136183, 30 to 24.934095, 40 fc rgb "#00FF00"
set object 44 rect from 0.156042, 30 to 25.975281, 40 fc rgb "#0000FF"
set object 45 rect from 0.160360, 30 to 28.875428, 40 fc rgb "#FFFF00"
set object 46 rect from 0.178340, 30 to 31.062181, 40 fc rgb "#FF00FF"
set object 47 rect from 0.191668, 30 to 34.549358, 40 fc rgb "#808080"
set object 48 rect from 0.213052, 30 to 35.286364, 40 fc rgb "#800080"
set object 49 rect from 0.220459, 30 to 36.598025, 40 fc rgb "#008080"
set object 50 rect from 0.225691, 30 to 145.240340, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.934035, 40 to 159.562438, 50 fc rgb "#FF0000"
set object 52 rect from 0.501898, 40 to 83.374730, 50 fc rgb "#00FF00"
set object 53 rect from 0.514284, 40 to 84.077865, 50 fc rgb "#0000FF"
set object 54 rect from 0.517075, 40 to 86.221303, 50 fc rgb "#FFFF00"
set object 55 rect from 0.530373, 40 to 87.643856, 50 fc rgb "#FF00FF"
set object 56 rect from 0.539052, 40 to 90.914458, 50 fc rgb "#808080"
set object 57 rect from 0.559075, 40 to 91.610917, 50 fc rgb "#800080"
set object 58 rect from 0.564983, 40 to 92.287021, 50 fc rgb "#008080"
set object 59 rect from 0.567504, 40 to 151.396595, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.955381, 50 to 162.722633, 60 fc rgb "#FF0000"
set object 61 rect from 0.707850, 50 to 116.784709, 60 fc rgb "#00FF00"
set object 62 rect from 0.719435, 50 to 117.405773, 60 fc rgb "#0000FF"
set object 63 rect from 0.721751, 50 to 119.409169, 60 fc rgb "#FFFF00"
set object 64 rect from 0.734044, 50 to 120.780754, 60 fc rgb "#FF00FF"
set object 65 rect from 0.742530, 50 to 124.019441, 60 fc rgb "#808080"
set object 66 rect from 0.762515, 50 to 124.700594, 60 fc rgb "#800080"
set object 67 rect from 0.840796, 50 to 137.419713, 60 fc rgb "#008080"
set object 68 rect from 0.844682, 50 to 154.999388, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
