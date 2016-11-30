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

set object 15 rect from 0.165148, 0 to 156.593416, 10 fc rgb "#FF0000"
set object 16 rect from 0.144441, 0 to 135.731471, 10 fc rgb "#00FF00"
set object 17 rect from 0.146332, 0 to 136.288468, 10 fc rgb "#0000FF"
set object 18 rect from 0.146685, 0 to 136.950553, 10 fc rgb "#FFFF00"
set object 19 rect from 0.147414, 0 to 137.922271, 10 fc rgb "#FF00FF"
set object 20 rect from 0.148441, 0 to 138.956301, 10 fc rgb "#808080"
set object 21 rect from 0.149565, 0 to 139.456585, 10 fc rgb "#800080"
set object 22 rect from 0.150374, 0 to 139.961511, 10 fc rgb "#008080"
set object 23 rect from 0.150638, 0 to 152.975245, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.163451, 10 to 155.320405, 20 fc rgb "#FF0000"
set object 25 rect from 0.129817, 10 to 122.221180, 20 fc rgb "#00FF00"
set object 26 rect from 0.131808, 10 to 122.859085, 20 fc rgb "#0000FF"
set object 27 rect from 0.132243, 10 to 123.557432, 20 fc rgb "#FFFF00"
set object 28 rect from 0.133054, 10 to 124.681653, 20 fc rgb "#FF00FF"
set object 29 rect from 0.134239, 10 to 125.763114, 20 fc rgb "#808080"
set object 30 rect from 0.135402, 10 to 126.353588, 20 fc rgb "#800080"
set object 31 rect from 0.136264, 10 to 126.904073, 20 fc rgb "#008080"
set object 32 rect from 0.136602, 10 to 151.378633, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.161757, 20 to 153.677291, 30 fc rgb "#FF0000"
set object 34 rect from 0.116677, 20 to 109.818387, 30 fc rgb "#00FF00"
set object 35 rect from 0.118477, 20 to 110.475815, 30 fc rgb "#0000FF"
set object 36 rect from 0.118930, 20 to 111.192757, 30 fc rgb "#FFFF00"
set object 37 rect from 0.119706, 20 to 112.409040, 30 fc rgb "#FF00FF"
set object 38 rect from 0.121006, 20 to 113.279412, 30 fc rgb "#808080"
set object 39 rect from 0.121940, 20 to 113.781553, 30 fc rgb "#800080"
set object 40 rect from 0.122765, 20 to 114.353432, 30 fc rgb "#008080"
set object 41 rect from 0.123101, 20 to 149.972642, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.160261, 30 to 152.368032, 40 fc rgb "#FF0000"
set object 43 rect from 0.102141, 30 to 96.441075, 40 fc rgb "#00FF00"
set object 44 rect from 0.104110, 30 to 97.086421, 40 fc rgb "#0000FF"
set object 45 rect from 0.104527, 30 to 97.917733, 40 fc rgb "#FFFF00"
set object 46 rect from 0.105433, 30 to 99.111707, 40 fc rgb "#FF00FF"
set object 47 rect from 0.106704, 30 to 99.967199, 40 fc rgb "#808080"
set object 48 rect from 0.107624, 30 to 100.500017, 40 fc rgb "#800080"
set object 49 rect from 0.108460, 30 to 101.051444, 40 fc rgb "#008080"
set object 50 rect from 0.108825, 30 to 148.510866, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.158700, 40 to 150.903443, 50 fc rgb "#FF0000"
set object 52 rect from 0.087261, 40 to 82.901964, 50 fc rgb "#00FF00"
set object 53 rect from 0.089540, 40 to 83.577987, 50 fc rgb "#0000FF"
set object 54 rect from 0.090021, 40 to 84.366525, 50 fc rgb "#FFFF00"
set object 55 rect from 0.090885, 40 to 85.577237, 50 fc rgb "#FF00FF"
set object 56 rect from 0.092174, 40 to 86.469005, 50 fc rgb "#808080"
set object 57 rect from 0.093122, 40 to 87.014834, 50 fc rgb "#800080"
set object 58 rect from 0.094007, 40 to 87.671334, 50 fc rgb "#008080"
set object 59 rect from 0.094439, 40 to 146.970982, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.156955, 50 to 151.519067, 60 fc rgb "#FF0000"
set object 61 rect from 0.067043, 50 to 64.851057, 60 fc rgb "#00FF00"
set object 62 rect from 0.070206, 50 to 65.838586, 60 fc rgb "#0000FF"
set object 63 rect from 0.070928, 50 to 67.910374, 60 fc rgb "#FFFF00"
set object 64 rect from 0.073197, 50 to 69.506986, 60 fc rgb "#FF00FF"
set object 65 rect from 0.074882, 50 to 70.607042, 60 fc rgb "#808080"
set object 66 rect from 0.076076, 50 to 71.257958, 60 fc rgb "#800080"
set object 67 rect from 0.077173, 50 to 72.424025, 60 fc rgb "#008080"
set object 68 rect from 0.078012, 50 to 145.018220, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
