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

set object 15 rect from 0.135648, 0 to 156.326600, 10 fc rgb "#FF0000"
set object 16 rect from 0.085683, 0 to 98.273411, 10 fc rgb "#00FF00"
set object 17 rect from 0.088119, 0 to 99.070916, 10 fc rgb "#0000FF"
set object 18 rect from 0.088563, 0 to 100.006192, 10 fc rgb "#FFFF00"
set object 19 rect from 0.089437, 0 to 101.359262, 10 fc rgb "#FF00FF"
set object 20 rect from 0.090637, 0 to 102.894906, 10 fc rgb "#808080"
set object 21 rect from 0.092000, 0 to 103.573682, 10 fc rgb "#800080"
set object 22 rect from 0.092871, 0 to 104.262537, 10 fc rgb "#008080"
set object 23 rect from 0.093200, 0 to 151.202184, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138783, 10 to 159.471815, 20 fc rgb "#FF0000"
set object 25 rect from 0.116767, 10 to 132.894299, 20 fc rgb "#00FF00"
set object 26 rect from 0.119014, 10 to 133.629079, 20 fc rgb "#0000FF"
set object 27 rect from 0.119415, 10 to 134.499390, 20 fc rgb "#FFFF00"
set object 28 rect from 0.120213, 10 to 135.700128, 20 fc rgb "#FF00FF"
set object 29 rect from 0.121277, 10 to 137.118163, 20 fc rgb "#808080"
set object 30 rect from 0.122576, 10 to 137.795817, 20 fc rgb "#800080"
set object 31 rect from 0.123415, 10 to 138.432029, 20 fc rgb "#008080"
set object 32 rect from 0.123711, 10 to 154.859281, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.133912, 20 to 154.331717, 30 fc rgb "#FF0000"
set object 34 rect from 0.069754, 20 to 80.209706, 30 fc rgb "#00FF00"
set object 35 rect from 0.072049, 20 to 80.964647, 30 fc rgb "#0000FF"
set object 36 rect from 0.072396, 20 to 82.002971, 30 fc rgb "#FFFF00"
set object 37 rect from 0.073338, 20 to 83.339239, 30 fc rgb "#FF00FF"
set object 38 rect from 0.074516, 20 to 84.734873, 30 fc rgb "#808080"
set object 39 rect from 0.075773, 20 to 85.433810, 30 fc rgb "#800080"
set object 40 rect from 0.076668, 20 to 86.129386, 30 fc rgb "#008080"
set object 41 rect from 0.077029, 20 to 149.404439, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.137207, 30 to 157.863365, 40 fc rgb "#FF0000"
set object 43 rect from 0.101085, 30 to 115.180062, 40 fc rgb "#00FF00"
set object 44 rect from 0.103187, 30 to 115.949565, 40 fc rgb "#0000FF"
set object 45 rect from 0.103632, 30 to 116.924044, 40 fc rgb "#FFFF00"
set object 46 rect from 0.104522, 30 to 118.150544, 40 fc rgb "#FF00FF"
set object 47 rect from 0.105613, 30 to 119.485692, 40 fc rgb "#808080"
set object 48 rect from 0.106802, 30 to 120.196950, 40 fc rgb "#800080"
set object 49 rect from 0.107680, 30 to 120.884685, 40 fc rgb "#008080"
set object 50 rect from 0.108041, 30 to 153.068256, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.132356, 40 to 152.575416, 50 fc rgb "#FF0000"
set object 52 rect from 0.052857, 40 to 61.869338, 50 fc rgb "#00FF00"
set object 53 rect from 0.055600, 40 to 62.663483, 50 fc rgb "#0000FF"
set object 54 rect from 0.056060, 40 to 63.671565, 50 fc rgb "#FFFF00"
set object 55 rect from 0.056961, 40 to 65.023514, 50 fc rgb "#FF00FF"
set object 56 rect from 0.058181, 40 to 66.406826, 50 fc rgb "#808080"
set object 57 rect from 0.059439, 40 to 67.041918, 50 fc rgb "#800080"
set object 58 rect from 0.060252, 40 to 67.810300, 50 fc rgb "#008080"
set object 59 rect from 0.060673, 40 to 147.564130, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.130612, 50 to 153.601420, 60 fc rgb "#FF0000"
set object 61 rect from 0.029448, 50 to 36.840908, 60 fc rgb "#00FF00"
set object 62 rect from 0.033492, 50 to 38.400075, 60 fc rgb "#0000FF"
set object 63 rect from 0.034428, 50 to 40.746665, 60 fc rgb "#FFFF00"
set object 64 rect from 0.036595, 50 to 42.873718, 60 fc rgb "#FF00FF"
set object 65 rect from 0.038399, 50 to 44.510170, 60 fc rgb "#808080"
set object 66 rect from 0.039876, 50 to 45.368160, 60 fc rgb "#800080"
set object 67 rect from 0.041627, 50 to 47.408965, 60 fc rgb "#008080"
set object 68 rect from 0.042523, 50 to 145.142493, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
