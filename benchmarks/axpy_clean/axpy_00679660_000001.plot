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

set object 15 rect from 0.974865, 0 to 159.640720, 10 fc rgb "#FF0000"
set object 16 rect from 0.826776, 0 to 132.494282, 10 fc rgb "#00FF00"
set object 17 rect from 0.839855, 0 to 133.080521, 10 fc rgb "#0000FF"
set object 18 rect from 0.842094, 0 to 133.745855, 10 fc rgb "#FFFF00"
set object 19 rect from 0.846270, 0 to 134.838610, 10 fc rgb "#FF00FF"
set object 20 rect from 0.853190, 0 to 138.119405, 10 fc rgb "#808080"
set object 21 rect from 0.873910, 0 to 138.661827, 10 fc rgb "#800080"
set object 22 rect from 0.878937, 0 to 139.206149, 10 fc rgb "#008080"
set object 23 rect from 0.880775, 0 to 153.666640, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.964812, 10 to 158.261014, 20 fc rgb "#FF0000"
set object 25 rect from 0.722436, 10 to 116.082713, 20 fc rgb "#00FF00"
set object 26 rect from 0.736132, 10 to 116.710083, 20 fc rgb "#0000FF"
set object 27 rect from 0.738560, 10 to 117.426511, 20 fc rgb "#FFFF00"
set object 28 rect from 0.743508, 10 to 118.698332, 20 fc rgb "#FF00FF"
set object 29 rect from 0.751320, 10 to 121.981975, 20 fc rgb "#808080"
set object 30 rect from 0.772158, 10 to 122.588781, 20 fc rgb "#800080"
set object 31 rect from 0.777536, 10 to 123.204284, 20 fc rgb "#008080"
set object 32 rect from 0.779638, 10 to 152.016908, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.923502, 20 to 155.253254, 30 fc rgb "#FF0000"
set object 34 rect from 0.239938, 20 to 40.861398, 30 fc rgb "#00FF00"
set object 35 rect from 0.261233, 20 to 41.866360, 30 fc rgb "#0000FF"
set object 36 rect from 0.265501, 20 to 44.575466, 30 fc rgb "#FFFF00"
set object 37 rect from 0.282825, 20 to 46.721745, 30 fc rgb "#FF00FF"
set object 38 rect from 0.296195, 20 to 50.127192, 30 fc rgb "#808080"
set object 39 rect from 0.317787, 20 to 50.928407, 30 fc rgb "#800080"
set object 40 rect from 0.325328, 20 to 52.286126, 30 fc rgb "#008080"
set object 41 rect from 0.331397, 20 to 145.064598, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.954742, 30 to 157.813663, 40 fc rgb "#FF0000"
set object 43 rect from 0.619454, 30 to 99.600121, 40 fc rgb "#00FF00"
set object 44 rect from 0.632134, 30 to 100.215151, 40 fc rgb "#0000FF"
set object 45 rect from 0.634265, 30 to 102.022910, 40 fc rgb "#FFFF00"
set object 46 rect from 0.645716, 30 to 103.355794, 40 fc rgb "#FF00FF"
set object 47 rect from 0.654125, 30 to 106.535822, 40 fc rgb "#808080"
set object 48 rect from 0.674254, 30 to 107.137565, 40 fc rgb "#800080"
set object 49 rect from 0.679658, 30 to 107.836753, 40 fc rgb "#008080"
set object 50 rect from 0.682594, 30 to 150.536913, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.934760, 40 to 155.174951, 50 fc rgb "#FF0000"
set object 52 rect from 0.386699, 40 to 63.194689, 50 fc rgb "#00FF00"
set object 53 rect from 0.401778, 40 to 63.817630, 50 fc rgb "#0000FF"
set object 54 rect from 0.404180, 40 to 65.758740, 50 fc rgb "#FFFF00"
set object 55 rect from 0.416678, 40 to 67.359905, 50 fc rgb "#FF00FF"
set object 56 rect from 0.426726, 40 to 70.622510, 50 fc rgb "#808080"
set object 57 rect from 0.447219, 40 to 71.279302, 50 fc rgb "#800080"
set object 58 rect from 0.453105, 40 to 72.072291, 50 fc rgb "#008080"
set object 59 rect from 0.456510, 40 to 147.268140, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.945245, 50 to 156.540260, 60 fc rgb "#FF0000"
set object 61 rect from 0.506948, 50 to 81.855198, 60 fc rgb "#00FF00"
set object 62 rect from 0.520007, 50 to 82.453935, 60 fc rgb "#0000FF"
set object 63 rect from 0.521985, 50 to 84.353600, 60 fc rgb "#FFFF00"
set object 64 rect from 0.534027, 50 to 85.867448, 60 fc rgb "#FF00FF"
set object 65 rect from 0.543577, 50 to 89.035297, 60 fc rgb "#808080"
set object 66 rect from 0.563709, 50 to 89.647640, 60 fc rgb "#800080"
set object 67 rect from 0.569232, 50 to 90.353628, 60 fc rgb "#008080"
set object 68 rect from 0.571955, 50 to 148.932264, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
