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

set object 15 rect from 0.140632, 0 to 159.249230, 10 fc rgb "#FF0000"
set object 16 rect from 0.119179, 0 to 133.623394, 10 fc rgb "#00FF00"
set object 17 rect from 0.121423, 0 to 134.429214, 10 fc rgb "#0000FF"
set object 18 rect from 0.121905, 0 to 135.267078, 10 fc rgb "#FFFF00"
set object 19 rect from 0.122651, 0 to 136.420670, 10 fc rgb "#FF00FF"
set object 20 rect from 0.123703, 0 to 137.885517, 10 fc rgb "#808080"
set object 21 rect from 0.125025, 0 to 138.455124, 10 fc rgb "#800080"
set object 22 rect from 0.125783, 0 to 139.002657, 10 fc rgb "#008080"
set object 23 rect from 0.126037, 0 to 154.662562, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.139036, 10 to 157.778791, 20 fc rgb "#FF0000"
set object 25 rect from 0.102722, 10 to 115.404663, 20 fc rgb "#00FF00"
set object 26 rect from 0.104908, 10 to 116.301020, 20 fc rgb "#0000FF"
set object 27 rect from 0.105469, 10 to 117.152142, 20 fc rgb "#FFFF00"
set object 28 rect from 0.106275, 10 to 118.443711, 20 fc rgb "#FF00FF"
set object 29 rect from 0.107428, 10 to 119.871021, 20 fc rgb "#808080"
set object 30 rect from 0.108727, 10 to 120.564295, 20 fc rgb "#800080"
set object 31 rect from 0.109592, 10 to 121.194601, 20 fc rgb "#008080"
set object 32 rect from 0.109907, 10 to 152.837803, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137372, 20 to 155.971732, 30 fc rgb "#FF0000"
set object 34 rect from 0.087350, 20 to 98.327384, 30 fc rgb "#00FF00"
set object 35 rect from 0.089437, 20 to 99.126625, 30 fc rgb "#0000FF"
set object 36 rect from 0.089927, 20 to 100.103551, 30 fc rgb "#FFFF00"
set object 37 rect from 0.090791, 20 to 101.495560, 30 fc rgb "#FF00FF"
set object 38 rect from 0.092066, 20 to 102.790453, 30 fc rgb "#808080"
set object 39 rect from 0.093234, 20 to 103.459414, 30 fc rgb "#800080"
set object 40 rect from 0.094084, 20 to 104.126172, 30 fc rgb "#008080"
set object 41 rect from 0.094463, 20 to 151.110195, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.135837, 30 to 154.385489, 40 fc rgb "#FF0000"
set object 43 rect from 0.070298, 30 to 79.512562, 40 fc rgb "#00FF00"
set object 44 rect from 0.072389, 30 to 80.330555, 40 fc rgb "#0000FF"
set object 45 rect from 0.072882, 30 to 81.350579, 40 fc rgb "#FFFF00"
set object 46 rect from 0.073808, 30 to 82.756932, 40 fc rgb "#FF00FF"
set object 47 rect from 0.075104, 30 to 84.077189, 40 fc rgb "#808080"
set object 48 rect from 0.076276, 30 to 84.726280, 40 fc rgb "#800080"
set object 49 rect from 0.077118, 30 to 85.369844, 40 fc rgb "#008080"
set object 50 rect from 0.077451, 30 to 149.339555, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.134250, 40 to 152.625837, 50 fc rgb "#FF0000"
set object 52 rect from 0.053869, 40 to 61.864590, 50 fc rgb "#00FF00"
set object 53 rect from 0.056406, 40 to 62.679260, 50 fc rgb "#0000FF"
set object 54 rect from 0.056889, 40 to 63.599897, 50 fc rgb "#FFFF00"
set object 55 rect from 0.057738, 40 to 65.066981, 50 fc rgb "#FF00FF"
set object 56 rect from 0.059077, 40 to 66.461194, 50 fc rgb "#808080"
set object 57 rect from 0.060329, 40 to 67.111404, 50 fc rgb "#800080"
set object 58 rect from 0.061211, 40 to 67.883010, 50 fc rgb "#008080"
set object 59 rect from 0.061621, 40 to 147.535786, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.132476, 50 to 153.412905, 60 fc rgb "#FF0000"
set object 61 rect from 0.030156, 50 to 36.830404, 60 fc rgb "#00FF00"
set object 62 rect from 0.033803, 50 to 38.128586, 60 fc rgb "#0000FF"
set object 63 rect from 0.034683, 50 to 40.356254, 60 fc rgb "#FFFF00"
set object 64 rect from 0.036729, 50 to 42.567375, 60 fc rgb "#FF00FF"
set object 65 rect from 0.038689, 50 to 44.086340, 60 fc rgb "#808080"
set object 66 rect from 0.040089, 50 to 44.992633, 60 fc rgb "#800080"
set object 67 rect from 0.041453, 50 to 46.777617, 60 fc rgb "#008080"
set object 68 rect from 0.042529, 50 to 145.193334, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
