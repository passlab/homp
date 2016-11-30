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

set object 15 rect from 0.131510, 0 to 154.377786, 10 fc rgb "#FF0000"
set object 16 rect from 0.068165, 0 to 80.552226, 10 fc rgb "#00FF00"
set object 17 rect from 0.070495, 0 to 81.364481, 10 fc rgb "#0000FF"
set object 18 rect from 0.070936, 0 to 82.205457, 10 fc rgb "#FFFF00"
set object 19 rect from 0.071687, 0 to 83.523216, 10 fc rgb "#FF00FF"
set object 20 rect from 0.072831, 0 to 84.056294, 10 fc rgb "#808080"
set object 21 rect from 0.073302, 0 to 84.743322, 10 fc rgb "#800080"
set object 22 rect from 0.074144, 0 to 85.425753, 10 fc rgb "#008080"
set object 23 rect from 0.074472, 0 to 150.260221, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.127831, 10 to 152.800385, 20 fc rgb "#FF0000"
set object 25 rect from 0.027801, 10 to 37.184490, 20 fc rgb "#00FF00"
set object 26 rect from 0.033101, 10 to 39.000862, 20 fc rgb "#0000FF"
set object 27 rect from 0.034145, 10 to 41.450264, 20 fc rgb "#FFFF00"
set object 28 rect from 0.036250, 10 to 43.113835, 20 fc rgb "#FF00FF"
set object 29 rect from 0.037658, 10 to 43.806606, 20 fc rgb "#808080"
set object 30 rect from 0.038257, 10 to 44.533844, 20 fc rgb "#800080"
set object 31 rect from 0.039407, 10 to 45.697654, 20 fc rgb "#008080"
set object 32 rect from 0.039931, 10 to 145.215509, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129744, 20 to 153.451795, 30 fc rgb "#FF0000"
set object 34 rect from 0.050046, 20 to 60.050551, 30 fc rgb "#00FF00"
set object 35 rect from 0.052645, 20 to 60.970799, 30 fc rgb "#0000FF"
set object 36 rect from 0.053198, 20 to 62.334514, 30 fc rgb "#FFFF00"
set object 37 rect from 0.054419, 20 to 64.026807, 30 fc rgb "#FF00FF"
set object 38 rect from 0.055886, 20 to 64.631115, 30 fc rgb "#808080"
set object 39 rect from 0.056386, 20 to 65.403159, 30 fc rgb "#800080"
set object 40 rect from 0.057348, 20 to 66.427956, 30 fc rgb "#008080"
set object 41 rect from 0.057968, 20 to 148.246242, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133213, 30 to 156.512396, 40 fc rgb "#FF0000"
set object 43 rect from 0.084186, 30 to 98.926186, 40 fc rgb "#00FF00"
set object 44 rect from 0.086510, 30 to 99.690187, 40 fc rgb "#0000FF"
set object 45 rect from 0.086918, 30 to 100.772427, 40 fc rgb "#FFFF00"
set object 46 rect from 0.087854, 30 to 102.124653, 40 fc rgb "#FF00FF"
set object 47 rect from 0.089024, 30 to 102.656581, 40 fc rgb "#808080"
set object 48 rect from 0.089480, 30 to 103.352800, 40 fc rgb "#800080"
set object 49 rect from 0.090358, 30 to 104.213309, 40 fc rgb "#008080"
set object 50 rect from 0.090828, 30 to 152.325899, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.136396, 40 to 159.940642, 50 fc rgb "#FF0000"
set object 52 rect from 0.114106, 40 to 133.133940, 50 fc rgb "#00FF00"
set object 53 rect from 0.116246, 40 to 133.935855, 50 fc rgb "#0000FF"
set object 54 rect from 0.116694, 40 to 134.891718, 50 fc rgb "#FFFF00"
set object 55 rect from 0.117528, 40 to 136.114121, 50 fc rgb "#FF00FF"
set object 56 rect from 0.118594, 40 to 136.564482, 50 fc rgb "#808080"
set object 57 rect from 0.118985, 40 to 137.191766, 50 fc rgb "#800080"
set object 58 rect from 0.119791, 40 to 137.868454, 50 fc rgb "#008080"
set object 59 rect from 0.120122, 40 to 156.105697, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.134799, 50 to 158.167929, 60 fc rgb "#FF0000"
set object 61 rect from 0.098736, 50 to 115.424031, 60 fc rgb "#00FF00"
set object 62 rect from 0.100838, 50 to 116.160459, 60 fc rgb "#0000FF"
set object 63 rect from 0.101222, 50 to 117.185256, 60 fc rgb "#FFFF00"
set object 64 rect from 0.102115, 50 to 118.397320, 60 fc rgb "#FF00FF"
set object 65 rect from 0.103170, 50 to 118.886741, 60 fc rgb "#808080"
set object 66 rect from 0.103601, 50 to 119.522067, 60 fc rgb "#800080"
set object 67 rect from 0.104412, 50 to 120.194160, 60 fc rgb "#008080"
set object 68 rect from 0.104737, 50 to 154.250263, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
