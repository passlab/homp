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

set object 15 rect from 0.140495, 0 to 159.845027, 10 fc rgb "#FF0000"
set object 16 rect from 0.118073, 0 to 133.120578, 10 fc rgb "#00FF00"
set object 17 rect from 0.120259, 0 to 133.834417, 10 fc rgb "#0000FF"
set object 18 rect from 0.120687, 0 to 134.679238, 10 fc rgb "#FFFF00"
set object 19 rect from 0.121447, 0 to 135.804917, 10 fc rgb "#FF00FF"
set object 20 rect from 0.122460, 0 to 137.208145, 10 fc rgb "#808080"
set object 21 rect from 0.123710, 0 to 137.785409, 10 fc rgb "#800080"
set object 22 rect from 0.124520, 0 to 138.404890, 10 fc rgb "#008080"
set object 23 rect from 0.124790, 0 to 155.357862, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138861, 10 to 158.361933, 20 fc rgb "#FF0000"
set object 25 rect from 0.101229, 10 to 114.446863, 20 fc rgb "#00FF00"
set object 26 rect from 0.103538, 10 to 115.305018, 20 fc rgb "#0000FF"
set object 27 rect from 0.103978, 10 to 116.165389, 20 fc rgb "#FFFF00"
set object 28 rect from 0.104782, 10 to 117.482001, 20 fc rgb "#FF00FF"
set object 29 rect from 0.105968, 10 to 118.929628, 20 fc rgb "#808080"
set object 30 rect from 0.107264, 10 to 119.611275, 20 fc rgb "#800080"
set object 31 rect from 0.108129, 10 to 120.258481, 20 fc rgb "#008080"
set object 32 rect from 0.108445, 10 to 153.394012, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137057, 20 to 156.381441, 30 fc rgb "#FF0000"
set object 34 rect from 0.085558, 20 to 97.025410, 30 fc rgb "#00FF00"
set object 35 rect from 0.087788, 20 to 97.833573, 30 fc rgb "#0000FF"
set object 36 rect from 0.088255, 20 to 98.750553, 30 fc rgb "#FFFF00"
set object 37 rect from 0.089070, 20 to 100.092739, 30 fc rgb "#FF00FF"
set object 38 rect from 0.090274, 20 to 101.391583, 30 fc rgb "#808080"
set object 39 rect from 0.091454, 20 to 102.093214, 30 fc rgb "#800080"
set object 40 rect from 0.092325, 20 to 102.722653, 30 fc rgb "#008080"
set object 41 rect from 0.092647, 20 to 151.634460, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.135551, 30 to 154.793898, 40 fc rgb "#FF0000"
set object 43 rect from 0.068825, 30 to 78.329462, 40 fc rgb "#00FF00"
set object 44 rect from 0.071026, 30 to 79.213158, 40 fc rgb "#0000FF"
set object 45 rect from 0.071470, 30 to 80.263337, 40 fc rgb "#FFFF00"
set object 46 rect from 0.072414, 30 to 81.578857, 40 fc rgb "#FF00FF"
set object 47 rect from 0.073613, 30 to 82.866618, 40 fc rgb "#808080"
set object 48 rect from 0.074761, 30 to 83.560473, 40 fc rgb "#800080"
set object 49 rect from 0.075632, 30 to 84.195471, 40 fc rgb "#008080"
set object 50 rect from 0.075977, 30 to 149.670610, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.133780, 40 to 152.905580, 50 fc rgb "#FF0000"
set object 52 rect from 0.053126, 40 to 61.239948, 50 fc rgb "#00FF00"
set object 53 rect from 0.055537, 40 to 61.985945, 50 fc rgb "#0000FF"
set object 54 rect from 0.055947, 40 to 62.993974, 50 fc rgb "#FFFF00"
set object 55 rect from 0.056857, 40 to 64.360544, 50 fc rgb "#FF00FF"
set object 56 rect from 0.058101, 40 to 65.775980, 50 fc rgb "#808080"
set object 57 rect from 0.059405, 40 to 66.482043, 50 fc rgb "#800080"
set object 58 rect from 0.060266, 40 to 67.232474, 50 fc rgb "#008080"
set object 59 rect from 0.060689, 40 to 147.800026, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.131834, 50 to 153.428420, 60 fc rgb "#FF0000"
set object 61 rect from 0.031018, 50 to 37.598312, 60 fc rgb "#00FF00"
set object 62 rect from 0.034443, 50 to 39.008190, 60 fc rgb "#0000FF"
set object 63 rect from 0.035290, 50 to 41.256239, 60 fc rgb "#FFFF00"
set object 64 rect from 0.037325, 50 to 43.342205, 60 fc rgb "#FF00FF"
set object 65 rect from 0.039169, 50 to 44.933024, 60 fc rgb "#808080"
set object 66 rect from 0.040619, 50 to 45.815595, 60 fc rgb "#800080"
set object 67 rect from 0.041873, 50 to 47.315397, 60 fc rgb "#008080"
set object 68 rect from 0.042744, 50 to 145.254445, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
