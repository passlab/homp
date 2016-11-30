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

set object 15 rect from 0.440310, 0 to 154.698999, 10 fc rgb "#FF0000"
set object 16 rect from 0.052708, 0 to 19.160812, 10 fc rgb "#00FF00"
set object 17 rect from 0.058281, 0 to 19.693237, 10 fc rgb "#0000FF"
set object 18 rect from 0.059499, 0 to 20.476455, 10 fc rgb "#FFFF00"
set object 19 rect from 0.061875, 0 to 21.085846, 10 fc rgb "#FF00FF"
set object 20 rect from 0.063693, 0 to 27.888004, 10 fc rgb "#808080"
set object 21 rect from 0.084296, 0 to 28.134481, 10 fc rgb "#800080"
set object 22 rect from 0.085389, 0 to 28.406831, 10 fc rgb "#008080"
set object 23 rect from 0.085771, 0 to 145.076799, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.464311, 10 to 161.701856, 20 fc rgb "#FF0000"
set object 25 rect from 0.394603, 10 to 131.731243, 20 fc rgb "#00FF00"
set object 26 rect from 0.397563, 10 to 132.092827, 20 fc rgb "#0000FF"
set object 27 rect from 0.398312, 10 to 132.355562, 20 fc rgb "#FFFF00"
set object 28 rect from 0.399137, 10 to 132.747666, 20 fc rgb "#FF00FF"
set object 29 rect from 0.400307, 10 to 139.504376, 20 fc rgb "#808080"
set object 30 rect from 0.420687, 10 to 139.707724, 20 fc rgb "#800080"
set object 31 rect from 0.421570, 10 to 139.907760, 20 fc rgb "#008080"
set object 32 rect from 0.421876, 10 to 153.762859, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.459643, 20 to 166.804216, 30 fc rgb "#FF0000"
set object 34 rect from 0.209463, 20 to 70.043117, 30 fc rgb "#00FF00"
set object 35 rect from 0.211515, 20 to 70.296228, 30 fc rgb "#0000FF"
set object 36 rect from 0.212026, 20 to 71.379329, 30 fc rgb "#FFFF00"
set object 37 rect from 0.215323, 20 to 77.804970, 30 fc rgb "#FF00FF"
set object 38 rect from 0.234663, 20 to 84.215682, 30 fc rgb "#808080"
set object 39 rect from 0.254007, 20 to 84.622714, 30 fc rgb "#800080"
set object 40 rect from 0.255517, 20 to 84.867203, 30 fc rgb "#008080"
set object 41 rect from 0.255957, 20 to 151.871325, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.462149, 30 to 187.866181, 40 fc rgb "#FF0000"
set object 43 rect from 0.265107, 30 to 88.447246, 40 fc rgb "#00FF00"
set object 44 rect from 0.267005, 30 to 88.687751, 40 fc rgb "#0000FF"
set object 45 rect from 0.267478, 30 to 89.928426, 40 fc rgb "#FFFF00"
set object 46 rect from 0.271220, 30 to 94.796919, 40 fc rgb "#FF00FF"
set object 47 rect from 0.285897, 30 to 122.543277, 40 fc rgb "#808080"
set object 48 rect from 0.369869, 30 to 123.369619, 40 fc rgb "#800080"
set object 49 rect from 0.372359, 30 to 123.809493, 40 fc rgb "#008080"
set object 50 rect from 0.373360, 30 to 152.912631, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.455097, 40 to 164.153690, 50 fc rgb "#FF0000"
set object 52 rect from 0.154490, 40 to 51.855934, 50 fc rgb "#00FF00"
set object 53 rect from 0.156696, 40 to 52.110706, 50 fc rgb "#0000FF"
set object 54 rect from 0.157226, 40 to 53.298969, 50 fc rgb "#FFFF00"
set object 55 rect from 0.160806, 40 to 58.322376, 50 fc rgb "#FF00FF"
set object 56 rect from 0.175951, 40 to 64.831947, 50 fc rgb "#808080"
set object 57 rect from 0.195567, 40 to 65.305989, 50 fc rgb "#800080"
set object 58 rect from 0.197314, 40 to 65.586633, 50 fc rgb "#008080"
set object 59 rect from 0.197857, 40 to 150.522507, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.449484, 50 to 162.291009, 60 fc rgb "#FF0000"
set object 61 rect from 0.100624, 50 to 34.070816, 60 fc rgb "#00FF00"
set object 62 rect from 0.103072, 50 to 34.439370, 60 fc rgb "#0000FF"
set object 63 rect from 0.103940, 50 to 35.600430, 60 fc rgb "#FFFF00"
set object 64 rect from 0.107482, 50 to 40.598959, 60 fc rgb "#FF00FF"
set object 65 rect from 0.122518, 50 to 47.068717, 60 fc rgb "#808080"
set object 66 rect from 0.142081, 50 to 47.538779, 60 fc rgb "#800080"
set object 67 rect from 0.143754, 50 to 48.033063, 60 fc rgb "#008080"
set object 68 rect from 0.144922, 50 to 148.544719, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
