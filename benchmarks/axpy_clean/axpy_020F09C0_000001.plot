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

set object 15 rect from 0.136696, 0 to 159.255755, 10 fc rgb "#FF0000"
set object 16 rect from 0.114255, 0 to 132.703449, 10 fc rgb "#00FF00"
set object 17 rect from 0.117025, 0 to 133.691255, 10 fc rgb "#0000FF"
set object 18 rect from 0.117444, 0 to 134.454613, 10 fc rgb "#FFFF00"
set object 19 rect from 0.118134, 0 to 135.498249, 10 fc rgb "#FF00FF"
set object 20 rect from 0.119030, 0 to 136.656958, 10 fc rgb "#808080"
set object 21 rect from 0.120054, 0 to 137.228904, 10 fc rgb "#800080"
set object 22 rect from 0.120821, 0 to 137.835034, 10 fc rgb "#008080"
set object 23 rect from 0.121091, 0 to 155.201990, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.135384, 10 to 157.735884, 20 fc rgb "#FF0000"
set object 25 rect from 0.075426, 10 to 87.443178, 20 fc rgb "#00FF00"
set object 26 rect from 0.077053, 10 to 88.137039, 20 fc rgb "#0000FF"
set object 27 rect from 0.077462, 10 to 88.896977, 20 fc rgb "#FFFF00"
set object 28 rect from 0.078161, 10 to 90.047711, 20 fc rgb "#FF00FF"
set object 29 rect from 0.079167, 10 to 91.143756, 20 fc rgb "#808080"
set object 30 rect from 0.080103, 10 to 91.707727, 20 fc rgb "#800080"
set object 31 rect from 0.080839, 10 to 92.310440, 20 fc rgb "#008080"
set object 32 rect from 0.081124, 10 to 153.549945, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.133707, 20 to 155.620120, 30 fc rgb "#FF0000"
set object 34 rect from 0.063557, 20 to 73.803005, 30 fc rgb "#00FF00"
set object 35 rect from 0.065086, 20 to 74.431919, 30 fc rgb "#0000FF"
set object 36 rect from 0.065427, 20 to 75.195277, 30 fc rgb "#FFFF00"
set object 37 rect from 0.066123, 20 to 76.399558, 30 fc rgb "#FF00FF"
set object 38 rect from 0.067154, 20 to 77.262040, 30 fc rgb "#808080"
set object 39 rect from 0.067923, 20 to 77.816898, 30 fc rgb "#800080"
set object 40 rect from 0.068652, 20 to 78.424165, 30 fc rgb "#008080"
set object 41 rect from 0.068936, 20 to 151.715609, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.132199, 30 to 153.842756, 40 fc rgb "#FF0000"
set object 43 rect from 0.050959, 30 to 59.541888, 40 fc rgb "#00FF00"
set object 44 rect from 0.052568, 30 to 60.107001, 40 fc rgb "#0000FF"
set object 45 rect from 0.052855, 30 to 60.931886, 40 fc rgb "#FFFF00"
set object 46 rect from 0.053591, 30 to 62.084894, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054600, 30 to 62.953072, 40 fc rgb "#808080"
set object 48 rect from 0.055364, 30 to 63.493122, 40 fc rgb "#800080"
set object 49 rect from 0.056051, 30 to 64.116340, 40 fc rgb "#008080"
set object 50 rect from 0.056377, 30 to 150.100027, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.130756, 40 to 152.338827, 50 fc rgb "#FF0000"
set object 52 rect from 0.037597, 40 to 44.708826, 50 fc rgb "#00FF00"
set object 53 rect from 0.039563, 40 to 45.360527, 50 fc rgb "#0000FF"
set object 54 rect from 0.039914, 40 to 46.143253, 50 fc rgb "#FFFF00"
set object 55 rect from 0.040608, 40 to 47.336142, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041657, 40 to 48.196345, 50 fc rgb "#808080"
set object 57 rect from 0.042426, 40 to 48.815008, 50 fc rgb "#800080"
set object 58 rect from 0.043198, 40 to 49.517978, 50 fc rgb "#008080"
set object 59 rect from 0.043573, 40 to 148.048077, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.128897, 50 to 152.100705, 60 fc rgb "#FF0000"
set object 61 rect from 0.020548, 50 to 25.957572, 60 fc rgb "#00FF00"
set object 62 rect from 0.023182, 50 to 27.092355, 60 fc rgb "#0000FF"
set object 63 rect from 0.023900, 50 to 28.632744, 60 fc rgb "#FFFF00"
set object 64 rect from 0.025284, 50 to 30.292760, 60 fc rgb "#FF00FF"
set object 65 rect from 0.026698, 50 to 31.385388, 60 fc rgb "#808080"
set object 66 rect from 0.027679, 50 to 32.119121, 60 fc rgb "#800080"
set object 67 rect from 0.028638, 50 to 33.222004, 60 fc rgb "#008080"
set object 68 rect from 0.029281, 50 to 145.521021, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
