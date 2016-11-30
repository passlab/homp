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

set object 15 rect from 6.012756, 0 to 159.451552, 10 fc rgb "#FF0000"
set object 16 rect from 4.754247, 0 to 114.083134, 10 fc rgb "#00FF00"
set object 17 rect from 4.756166, 0 to 114.099566, 10 fc rgb "#0000FF"
set object 18 rect from 4.756649, 0 to 114.115278, 10 fc rgb "#FFFF00"
set object 19 rect from 4.757334, 0 to 114.139386, 10 fc rgb "#FF00FF"
set object 20 rect from 4.758341, 0 to 129.300972, 10 fc rgb "#808080"
set object 21 rect from 5.390506, 0 to 129.320690, 10 fc rgb "#800080"
set object 22 rect from 5.391448, 0 to 129.334939, 10 fc rgb "#008080"
set object 23 rect from 5.391778, 0 to 144.215339, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 6.015930, 10 to 158.418634, 20 fc rgb "#FF0000"
set object 25 rect from 5.402878, 10 to 129.641024, 20 fc rgb "#00FF00"
set object 26 rect from 5.404737, 10 to 129.653978, 20 fc rgb "#0000FF"
set object 27 rect from 5.405075, 10 to 129.668490, 20 fc rgb "#FFFF00"
set object 28 rect from 5.405683, 10 to 129.688448, 20 fc rgb "#FF00FF"
set object 29 rect from 5.406526, 10 to 143.750286, 20 fc rgb "#808080"
set object 30 rect from 5.993040, 10 to 143.772379, 20 fc rgb "#800080"
set object 31 rect from 5.993954, 10 to 143.785021, 20 fc rgb "#008080"
set object 32 rect from 5.994177, 10 to 144.294883, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 6.008922, 20 to 171.455326, 30 fc rgb "#FF0000"
set object 34 rect from 1.279869, 20 to 30.746679, 30 fc rgb "#00FF00"
set object 35 rect from 1.282109, 20 to 30.762966, 30 fc rgb "#0000FF"
set object 36 rect from 1.282535, 20 to 30.865802, 30 fc rgb "#FFFF00"
set object 37 rect from 1.286875, 20 to 40.877743, 30 fc rgb "#FF00FF"
set object 38 rect from 1.704215, 20 to 56.995966, 30 fc rgb "#808080"
set object 39 rect from 2.376206, 20 to 58.079666, 30 fc rgb "#800080"
set object 40 rect from 2.421580, 20 to 58.108092, 30 fc rgb "#008080"
set object 41 rect from 2.422505, 20 to 144.124593, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 6.006265, 30 to 172.689045, 40 fc rgb "#FF0000"
set object 43 rect from 0.026340, 30 to 0.692290, 40 fc rgb "#00FF00"
set object 44 rect from 0.029322, 30 to 0.716494, 40 fc rgb "#0000FF"
set object 45 rect from 0.029976, 30 to 0.892853, 40 fc rgb "#FFFF00"
set object 46 rect from 0.037354, 30 to 13.221350, 40 fc rgb "#FF00FF"
set object 47 rect from 0.551273, 30 to 28.256520, 40 fc rgb "#808080"
set object 48 rect from 1.178193, 30 to 29.326596, 40 fc rgb "#800080"
set object 49 rect from 1.223186, 30 to 30.429895, 40 fc rgb "#008080"
set object 50 rect from 1.268698, 30 to 144.049463, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 6.014394, 40 to 171.646006, 50 fc rgb "#FF0000"
set object 52 rect from 3.599696, 40 to 86.382436, 50 fc rgb "#00FF00"
set object 53 rect from 3.601418, 40 to 86.396157, 50 fc rgb "#0000FF"
set object 54 rect from 3.601755, 40 to 86.481002, 50 fc rgb "#FFFF00"
set object 55 rect from 3.605308, 40 to 98.504086, 50 fc rgb "#FF00FF"
set object 56 rect from 4.106520, 40 to 112.786254, 50 fc rgb "#808080"
set object 57 rect from 4.702027, 40 to 113.774530, 50 fc rgb "#800080"
set object 58 rect from 4.743374, 40 to 113.795279, 50 fc rgb "#008080"
set object 59 rect from 4.744043, 40 to 144.259357, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 6.010824, 50 to 171.600237, 60 fc rgb "#FF0000"
set object 61 rect from 2.438758, 50 to 58.545007, 60 fc rgb "#00FF00"
set object 62 rect from 2.440919, 50 to 58.561319, 60 fc rgb "#0000FF"
set object 63 rect from 2.441388, 50 to 58.660677, 60 fc rgb "#FFFF00"
set object 64 rect from 2.445580, 50 to 69.815256, 60 fc rgb "#FF00FF"
set object 65 rect from 2.910565, 50 to 84.860357, 60 fc rgb "#808080"
set object 66 rect from 3.537847, 50 to 85.977593, 60 fc rgb "#800080"
set object 67 rect from 3.584843, 50 to 86.006786, 60 fc rgb "#008080"
set object 68 rect from 3.585594, 50 to 144.170242, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
