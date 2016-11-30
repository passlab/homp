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

set object 15 rect from 1.065727, 0 to 159.880326, 10 fc rgb "#FF0000"
set object 16 rect from 0.804604, 0 to 130.732070, 10 fc rgb "#00FF00"
set object 17 rect from 0.904650, 0 to 131.593344, 10 fc rgb "#0000FF"
set object 18 rect from 0.908011, 0 to 132.312038, 10 fc rgb "#FFFF00"
set object 19 rect from 0.912966, 0 to 133.294714, 10 fc rgb "#FF00FF"
set object 20 rect from 0.919832, 0 to 136.319757, 10 fc rgb "#808080"
set object 21 rect from 0.940689, 0 to 136.868315, 10 fc rgb "#800080"
set object 22 rect from 0.946184, 0 to 137.385398, 10 fc rgb "#008080"
set object 23 rect from 0.947964, 0 to 154.083770, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.044646, 10 to 156.695291, 20 fc rgb "#FF0000"
set object 25 rect from 0.591540, 10 to 87.416180, 20 fc rgb "#00FF00"
set object 26 rect from 0.605172, 10 to 88.072217, 20 fc rgb "#0000FF"
set object 27 rect from 0.607963, 10 to 88.694310, 20 fc rgb "#FFFF00"
set object 28 rect from 0.612524, 10 to 89.824786, 20 fc rgb "#FF00FF"
set object 29 rect from 0.620120, 10 to 92.803852, 20 fc rgb "#808080"
set object 30 rect from 0.640869, 10 to 93.372134, 20 fc rgb "#800080"
set object 31 rect from 0.646244, 10 to 93.927075, 20 fc rgb "#008080"
set object 32 rect from 0.648343, 10 to 150.846670, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.021090, 20 to 154.921834, 30 fc rgb "#FF0000"
set object 34 rect from 0.360355, 20 to 54.127494, 30 fc rgb "#00FF00"
set object 35 rect from 0.375581, 20 to 54.816744, 30 fc rgb "#0000FF"
set object 36 rect from 0.378999, 20 to 56.620081, 30 fc rgb "#FFFF00"
set object 37 rect from 0.391283, 20 to 58.157985, 30 fc rgb "#FF00FF"
set object 38 rect from 0.401711, 20 to 61.134005, 30 fc rgb "#808080"
set object 39 rect from 0.422253, 20 to 61.690394, 30 fc rgb "#800080"
set object 40 rect from 0.427954, 20 to 62.279274, 30 fc rgb "#008080"
set object 41 rect from 0.430244, 20 to 147.440885, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.032489, 30 to 155.995746, 40 fc rgb "#FF0000"
set object 43 rect from 0.477021, 30 to 70.729412, 40 fc rgb "#00FF00"
set object 44 rect from 0.490039, 30 to 71.292909, 40 fc rgb "#0000FF"
set object 45 rect from 0.492329, 30 to 73.007479, 40 fc rgb "#FFFF00"
set object 46 rect from 0.504157, 30 to 74.178130, 40 fc rgb "#FF00FF"
set object 47 rect from 0.512218, 30 to 77.076841, 40 fc rgb "#808080"
set object 48 rect from 0.532220, 30 to 77.676744, 40 fc rgb "#800080"
set object 49 rect from 0.538314, 30 to 78.309138, 40 fc rgb "#008080"
set object 50 rect from 0.540800, 30 to 149.221595, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.056052, 40 to 159.648539, 50 fc rgb "#FF0000"
set object 52 rect from 0.689092, 40 to 101.300982, 50 fc rgb "#00FF00"
set object 53 rect from 0.700910, 40 to 101.881594, 50 fc rgb "#0000FF"
set object 54 rect from 0.703308, 40 to 103.717710, 50 fc rgb "#FFFF00"
set object 55 rect from 0.715912, 40 to 105.008750, 50 fc rgb "#FF00FF"
set object 56 rect from 0.724801, 40 to 107.951554, 50 fc rgb "#808080"
set object 57 rect from 0.745090, 40 to 108.515196, 50 fc rgb "#800080"
set object 58 rect from 0.750682, 40 to 109.138887, 50 fc rgb "#008080"
set object 59 rect from 0.753202, 40 to 152.585461, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.008355, 50 to 156.237822, 60 fc rgb "#FF0000"
set object 61 rect from 0.183348, 50 to 30.151958, 60 fc rgb "#00FF00"
set object 62 rect from 0.211844, 50 to 31.382805, 60 fc rgb "#0000FF"
set object 63 rect from 0.217316, 50 to 34.937985, 60 fc rgb "#FFFF00"
set object 64 rect from 0.242668, 50 to 37.387784, 60 fc rgb "#FF00FF"
set object 65 rect from 0.258613, 50 to 40.543659, 60 fc rgb "#808080"
set object 66 rect from 0.280412, 50 to 41.236244, 60 fc rgb "#800080"
set object 67 rect from 0.287969, 50 to 42.337567, 60 fc rgb "#008080"
set object 68 rect from 0.292827, 50 to 145.355004, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
