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

set object 15 rect from 0.464298, 0 to 154.221182, 10 fc rgb "#FF0000"
set object 16 rect from 0.118027, 0 to 39.056385, 10 fc rgb "#00FF00"
set object 17 rect from 0.125967, 0 to 40.021269, 10 fc rgb "#0000FF"
set object 18 rect from 0.128376, 0 to 41.365309, 10 fc rgb "#FFFF00"
set object 19 rect from 0.132685, 0 to 41.956036, 10 fc rgb "#FF00FF"
set object 20 rect from 0.134574, 0 to 48.519266, 10 fc rgb "#808080"
set object 21 rect from 0.155868, 0 to 48.845680, 10 fc rgb "#800080"
set object 22 rect from 0.157121, 0 to 49.095639, 10 fc rgb "#008080"
set object 23 rect from 0.157448, 0 to 144.476513, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.471900, 10 to 152.963898, 20 fc rgb "#FF0000"
set object 25 rect from 0.432542, 10 to 135.429296, 20 fc rgb "#00FF00"
set object 26 rect from 0.434299, 10 to 135.630573, 20 fc rgb "#0000FF"
set object 27 rect from 0.434734, 10 to 135.840587, 20 fc rgb "#FFFF00"
set object 28 rect from 0.435442, 10 to 136.143910, 20 fc rgb "#FF00FF"
set object 29 rect from 0.436405, 10 to 141.179298, 20 fc rgb "#808080"
set object 30 rect from 0.452537, 10 to 141.352488, 20 fc rgb "#800080"
set object 31 rect from 0.453298, 10 to 141.504461, 20 fc rgb "#008080"
set object 32 rect from 0.453572, 10 to 147.101244, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.469199, 20 to 156.134107, 30 fc rgb "#FF0000"
set object 34 rect from 0.342908, 20 to 107.436029, 30 fc rgb "#00FF00"
set object 35 rect from 0.344610, 20 to 107.633874, 30 fc rgb "#0000FF"
set object 36 rect from 0.345020, 20 to 108.455839, 30 fc rgb "#FFFF00"
set object 37 rect from 0.347654, 20 to 112.132522, 30 fc rgb "#FF00FF"
set object 38 rect from 0.359436, 20 to 117.054007, 30 fc rgb "#808080"
set object 39 rect from 0.375207, 20 to 117.353581, 30 fc rgb "#800080"
set object 40 rect from 0.376436, 20 to 117.556421, 30 fc rgb "#008080"
set object 41 rect from 0.376819, 20 to 146.272104, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.466219, 30 to 172.719115, 40 fc rgb "#FF0000"
set object 43 rect from 0.188174, 30 to 59.430100, 40 fc rgb "#00FF00"
set object 44 rect from 0.191145, 30 to 60.068886, 40 fc rgb "#0000FF"
set object 45 rect from 0.192604, 30 to 61.332726, 40 fc rgb "#FFFF00"
set object 46 rect from 0.196685, 30 to 81.572570, 40 fc rgb "#FF00FF"
set object 47 rect from 0.261613, 30 to 86.677858, 40 fc rgb "#808080"
set object 48 rect from 0.277888, 30 to 87.032665, 40 fc rgb "#800080"
set object 49 rect from 0.279303, 30 to 87.734487, 40 fc rgb "#008080"
set object 50 rect from 0.281274, 30 to 145.280692, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.470568, 40 to 156.607816, 50 fc rgb "#FF0000"
set object 52 rect from 0.390246, 40 to 122.139634, 50 fc rgb "#00FF00"
set object 53 rect from 0.391715, 40 to 122.340599, 50 fc rgb "#0000FF"
set object 54 rect from 0.392146, 40 to 123.162876, 50 fc rgb "#FFFF00"
set object 55 rect from 0.394787, 40 to 126.874192, 50 fc rgb "#FF00FF"
set object 56 rect from 0.406681, 40 to 131.811283, 50 fc rgb "#808080"
set object 57 rect from 0.422505, 40 to 132.104305, 50 fc rgb "#800080"
set object 58 rect from 0.423685, 40 to 132.288107, 50 fc rgb "#008080"
set object 59 rect from 0.424026, 40 to 146.684642, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.467730, 50 to 156.128806, 60 fc rgb "#FF0000"
set object 61 rect from 0.297404, 50 to 93.322533, 60 fc rgb "#00FF00"
set object 62 rect from 0.299373, 50 to 93.574049, 60 fc rgb "#0000FF"
set object 63 rect from 0.299964, 50 to 94.485890, 60 fc rgb "#FFFF00"
set object 64 rect from 0.302901, 50 to 98.459027, 60 fc rgb "#FF00FF"
set object 65 rect from 0.315640, 50 to 103.407970, 60 fc rgb "#808080"
set object 66 rect from 0.331490, 50 to 103.704429, 60 fc rgb "#800080"
set object 67 rect from 0.332672, 50 to 103.907892, 60 fc rgb "#008080"
set object 68 rect from 0.333100, 50 to 145.775303, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
