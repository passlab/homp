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

set object 15 rect from 0.819449, 0 to 155.351637, 10 fc rgb "#FF0000"
set object 16 rect from 0.272132, 0 to 51.849438, 10 fc rgb "#00FF00"
set object 17 rect from 0.288730, 0 to 52.672029, 10 fc rgb "#0000FF"
set object 18 rect from 0.291606, 0 to 53.580284, 10 fc rgb "#FFFF00"
set object 19 rect from 0.296869, 0 to 55.100319, 10 fc rgb "#FF00FF"
set object 20 rect from 0.305033, 0 to 59.062406, 10 fc rgb "#808080"
set object 21 rect from 0.326957, 0 to 59.718196, 10 fc rgb "#800080"
set object 22 rect from 0.332284, 0 to 60.446068, 10 fc rgb "#008080"
set object 23 rect from 0.334707, 0 to 147.566188, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.844085, 10 to 159.449736, 20 fc rgb "#FF0000"
set object 25 rect from 0.601207, 10 to 110.886311, 20 fc rgb "#00FF00"
set object 26 rect from 0.614717, 10 to 111.641346, 20 fc rgb "#0000FF"
set object 27 rect from 0.617196, 10 to 112.503601, 20 fc rgb "#FFFF00"
set object 28 rect from 0.622214, 10 to 113.827132, 20 fc rgb "#FF00FF"
set object 29 rect from 0.629392, 10 to 117.693053, 20 fc rgb "#808080"
set object 30 rect from 0.650900, 10 to 118.413137, 20 fc rgb "#800080"
set object 31 rect from 0.656298, 10 to 119.099171, 20 fc rgb "#008080"
set object 32 rect from 0.658444, 10 to 151.993722, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.942484, 20 to 178.477610, 30 fc rgb "#FF0000"
set object 34 rect from 0.494823, 20 to 91.419972, 30 fc rgb "#00FF00"
set object 35 rect from 0.507385, 20 to 92.146938, 30 fc rgb "#0000FF"
set object 36 rect from 0.509558, 20 to 94.221343, 30 fc rgb "#FFFF00"
set object 37 rect from 0.521001, 20 to 95.736489, 30 fc rgb "#FF00FF"
set object 38 rect from 0.529364, 20 to 99.387071, 30 fc rgb "#808080"
set object 39 rect from 0.549544, 20 to 100.091581, 30 fc rgb "#800080"
set object 40 rect from 0.555108, 20 to 100.817459, 30 fc rgb "#008080"
set object 41 rect from 0.557589, 20 to 169.989284, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.831343, 30 to 158.864215, 40 fc rgb "#FF0000"
set object 43 rect from 0.378365, 30 to 70.784951, 40 fc rgb "#00FF00"
set object 44 rect from 0.393361, 30 to 71.574578, 40 fc rgb "#0000FF"
set object 45 rect from 0.395958, 30 to 73.742072, 40 fc rgb "#FFFF00"
set object 46 rect from 0.408066, 30 to 75.459878, 40 fc rgb "#FF00FF"
set object 47 rect from 0.417575, 30 to 79.166785, 40 fc rgb "#808080"
set object 48 rect from 0.438042, 30 to 79.981950, 40 fc rgb "#800080"
set object 49 rect from 0.444160, 30 to 80.795846, 40 fc rgb "#008080"
set object 50 rect from 0.446887, 30 to 149.537001, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.953589, 40 to 180.540785, 50 fc rgb "#FF0000"
set object 52 rect from 0.704103, 40 to 129.335911, 50 fc rgb "#00FF00"
set object 53 rect from 0.716486, 40 to 129.985361, 50 fc rgb "#0000FF"
set object 54 rect from 0.718467, 40 to 132.108845, 50 fc rgb "#FFFF00"
set object 55 rect from 0.730278, 40 to 133.642463, 50 fc rgb "#FF00FF"
set object 56 rect from 0.738762, 40 to 137.320214, 50 fc rgb "#808080"
set object 57 rect from 0.759066, 40 to 138.033054, 50 fc rgb "#800080"
set object 58 rect from 0.764729, 40 to 138.788634, 50 fc rgb "#008080"
set object 59 rect from 0.767137, 40 to 172.109872, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.807708, 50 to 156.886880, 60 fc rgb "#FF0000"
set object 61 rect from 0.128320, 50 to 26.359286, 60 fc rgb "#00FF00"
set object 62 rect from 0.148769, 50 to 27.680103, 60 fc rgb "#0000FF"
set object 63 rect from 0.153681, 50 to 30.808374, 60 fc rgb "#FFFF00"
set object 64 rect from 0.171260, 50 to 33.307294, 60 fc rgb "#FF00FF"
set object 65 rect from 0.184743, 50 to 37.249459, 60 fc rgb "#808080"
set object 66 rect from 0.206679, 50 to 38.098128, 60 fc rgb "#800080"
set object 67 rect from 0.213959, 50 to 39.435065, 60 fc rgb "#008080"
set object 68 rect from 0.218734, 50 to 145.219397, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
