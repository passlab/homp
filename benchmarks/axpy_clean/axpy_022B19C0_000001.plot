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

set object 15 rect from 0.164637, 0 to 153.888260, 10 fc rgb "#FF0000"
set object 16 rect from 0.115707, 0 to 107.472769, 10 fc rgb "#00FF00"
set object 17 rect from 0.117959, 0 to 108.158374, 10 fc rgb "#0000FF"
set object 18 rect from 0.118429, 0 to 108.899743, 10 fc rgb "#FFFF00"
set object 19 rect from 0.119242, 0 to 109.820282, 10 fc rgb "#FF00FF"
set object 20 rect from 0.120273, 0 to 111.060771, 10 fc rgb "#808080"
set object 21 rect from 0.121604, 0 to 111.552579, 10 fc rgb "#800080"
set object 22 rect from 0.122396, 0 to 112.054442, 10 fc rgb "#008080"
set object 23 rect from 0.122694, 0 to 149.987621, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.162975, 10 to 152.601148, 20 fc rgb "#FF0000"
set object 25 rect from 0.099927, 10 to 93.096994, 20 fc rgb "#00FF00"
set object 26 rect from 0.102201, 10 to 93.892296, 20 fc rgb "#0000FF"
set object 27 rect from 0.102843, 10 to 94.640977, 20 fc rgb "#FFFF00"
set object 28 rect from 0.103673, 10 to 95.724234, 20 fc rgb "#FF00FF"
set object 29 rect from 0.104861, 10 to 96.922672, 20 fc rgb "#808080"
set object 30 rect from 0.106171, 10 to 97.462015, 20 fc rgb "#800080"
set object 31 rect from 0.107003, 10 to 97.997702, 20 fc rgb "#008080"
set object 32 rect from 0.107321, 10 to 148.373249, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.161258, 20 to 151.151322, 30 fc rgb "#FF0000"
set object 34 rect from 0.084286, 20 to 78.948840, 30 fc rgb "#00FF00"
set object 35 rect from 0.086736, 20 to 79.638102, 30 fc rgb "#0000FF"
set object 36 rect from 0.087231, 20 to 80.416949, 30 fc rgb "#FFFF00"
set object 37 rect from 0.088084, 20 to 81.664752, 30 fc rgb "#FF00FF"
set object 38 rect from 0.089460, 20 to 82.755321, 30 fc rgb "#808080"
set object 39 rect from 0.090640, 20 to 83.335801, 30 fc rgb "#800080"
set object 40 rect from 0.091554, 20 to 83.918108, 30 fc rgb "#008080"
set object 41 rect from 0.091946, 20 to 146.858517, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.159473, 30 to 152.346102, 40 fc rgb "#FF0000"
set object 43 rect from 0.062359, 30 to 59.712579, 40 fc rgb "#00FF00"
set object 44 rect from 0.065813, 30 to 61.143208, 40 fc rgb "#0000FF"
set object 45 rect from 0.067025, 30 to 63.311550, 40 fc rgb "#FFFF00"
set object 46 rect from 0.069410, 30 to 65.037448, 40 fc rgb "#FF00FF"
set object 47 rect from 0.071278, 30 to 66.397689, 40 fc rgb "#808080"
set object 48 rect from 0.072795, 30 to 67.141799, 40 fc rgb "#800080"
set object 49 rect from 0.074036, 30 to 68.317384, 40 fc rgb "#008080"
set object 50 rect from 0.074868, 30 to 144.985443, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.167718, 40 to 156.809851, 50 fc rgb "#FF0000"
set object 52 rect from 0.145670, 40 to 134.788205, 50 fc rgb "#00FF00"
set object 53 rect from 0.147800, 40 to 135.435416, 50 fc rgb "#0000FF"
set object 54 rect from 0.148268, 40 to 136.226148, 50 fc rgb "#FFFF00"
set object 55 rect from 0.149145, 40 to 137.305748, 50 fc rgb "#FF00FF"
set object 56 rect from 0.150312, 40 to 138.364323, 50 fc rgb "#808080"
set object 57 rect from 0.151480, 40 to 138.924691, 50 fc rgb "#800080"
set object 58 rect from 0.152347, 40 to 139.469519, 50 fc rgb "#008080"
set object 59 rect from 0.152705, 40 to 152.889103, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.166215, 50 to 155.581247, 60 fc rgb "#FF0000"
set object 61 rect from 0.130446, 50 to 120.833848, 60 fc rgb "#00FF00"
set object 62 rect from 0.132551, 50 to 121.577959, 60 fc rgb "#0000FF"
set object 63 rect from 0.133124, 50 to 122.398857, 60 fc rgb "#FFFF00"
set object 64 rect from 0.134026, 50 to 123.484856, 60 fc rgb "#FF00FF"
set object 65 rect from 0.135212, 50 to 124.548001, 60 fc rgb "#808080"
set object 66 rect from 0.136376, 50 to 125.157733, 60 fc rgb "#800080"
set object 67 rect from 0.137292, 50 to 125.747354, 60 fc rgb "#008080"
set object 68 rect from 0.137678, 50 to 151.434705, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
