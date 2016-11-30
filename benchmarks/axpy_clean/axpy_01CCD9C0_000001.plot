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

set object 15 rect from 1.638085, 0 to 145.603759, 10 fc rgb "#FF0000"
set object 16 rect from 1.610671, 0 to 143.018306, 10 fc rgb "#00FF00"
set object 17 rect from 1.612963, 0 to 143.085534, 10 fc rgb "#0000FF"
set object 18 rect from 1.613461, 0 to 143.152582, 10 fc rgb "#FFFF00"
set object 19 rect from 1.614228, 0 to 143.246326, 10 fc rgb "#FF00FF"
set object 20 rect from 1.615276, 0 to 143.359848, 10 fc rgb "#808080"
set object 21 rect from 1.616560, 0 to 143.407208, 10 fc rgb "#800080"
set object 22 rect from 1.617358, 0 to 143.454832, 10 fc rgb "#008080"
set object 23 rect from 1.617628, 0 to 145.231176, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.636444, 10 to 145.589829, 20 fc rgb "#FF0000"
set object 25 rect from 1.590119, 10 to 141.290387, 20 fc rgb "#00FF00"
set object 26 rect from 1.593487, 10 to 141.376505, 20 fc rgb "#0000FF"
set object 27 rect from 1.594203, 10 to 141.471134, 20 fc rgb "#FFFF00"
set object 28 rect from 1.595290, 10 to 141.589002, 20 fc rgb "#FF00FF"
set object 29 rect from 1.596608, 10 to 141.749617, 20 fc rgb "#808080"
set object 30 rect from 1.598422, 10 to 141.817553, 20 fc rgb "#800080"
set object 31 rect from 1.599567, 10 to 141.883980, 20 fc rgb "#008080"
set object 32 rect from 1.599941, 10 to 145.074552, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.634635, 20 to 145.307623, 30 fc rgb "#FF0000"
set object 34 rect from 0.071814, 20 to 6.530761, 30 fc rgb "#00FF00"
set object 35 rect from 0.074003, 20 to 6.588675, 30 fc rgb "#0000FF"
set object 36 rect from 0.074404, 20 to 6.660068, 30 fc rgb "#FFFF00"
set object 37 rect from 0.075214, 20 to 6.767469, 30 fc rgb "#FF00FF"
set object 38 rect from 0.076419, 20 to 6.873186, 30 fc rgb "#808080"
set object 39 rect from 0.077617, 20 to 6.927198, 30 fc rgb "#800080"
set object 40 rect from 0.078496, 20 to 6.981832, 30 fc rgb "#008080"
set object 41 rect from 1.574325, 20 to 144.876688, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.632436, 30 to 145.111709, 40 fc rgb "#FF0000"
set object 43 rect from 0.056500, 30 to 5.176130, 40 fc rgb "#00FF00"
set object 44 rect from 0.058827, 30 to 5.238569, 40 fc rgb "#0000FF"
set object 45 rect from 0.059183, 30 to 5.319541, 40 fc rgb "#FFFF00"
set object 46 rect from 0.060095, 30 to 5.421975, 40 fc rgb "#FF00FF"
set object 47 rect from 0.061264, 30 to 5.529997, 40 fc rgb "#808080"
set object 48 rect from 0.062480, 30 to 5.583478, 40 fc rgb "#800080"
set object 49 rect from 0.063348, 30 to 5.639084, 40 fc rgb "#008080"
set object 50 rect from 0.063701, 30 to 144.702680, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.629887, 40 to 144.893802, 50 fc rgb "#FF0000"
set object 52 rect from 0.041249, 40 to 3.854938, 50 fc rgb "#00FF00"
set object 53 rect from 0.043820, 40 to 3.910988, 50 fc rgb "#0000FF"
set object 54 rect from 0.044213, 40 to 3.988413, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045112, 40 to 4.095904, 50 fc rgb "#FF00FF"
set object 56 rect from 0.046297, 40 to 4.202950, 50 fc rgb "#808080"
set object 57 rect from 0.047509, 40 to 4.259357, 50 fc rgb "#800080"
set object 58 rect from 0.048439, 40 to 4.320641, 50 fc rgb "#008080"
set object 59 rect from 0.048852, 40 to 144.461981, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.626899, 50 to 144.939742, 60 fc rgb "#FF0000"
set object 61 rect from 0.019394, 50 to 1.978020, 60 fc rgb "#00FF00"
set object 62 rect from 0.022829, 50 to 2.154512, 60 fc rgb "#0000FF"
set object 63 rect from 0.024421, 50 to 2.335968, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026542, 50 to 2.513967, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028486, 50 to 2.645314, 60 fc rgb "#808080"
set object 66 rect from 0.029973, 50 to 2.719102, 60 fc rgb "#800080"
set object 67 rect from 0.031328, 50 to 2.840784, 60 fc rgb "#008080"
set object 68 rect from 0.032167, 50 to 144.152101, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
