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

set object 15 rect from 0.137881, 0 to 158.587253, 10 fc rgb "#FF0000"
set object 16 rect from 0.117600, 0 to 133.898009, 10 fc rgb "#00FF00"
set object 17 rect from 0.119410, 0 to 134.644800, 10 fc rgb "#0000FF"
set object 18 rect from 0.119832, 0 to 135.325242, 10 fc rgb "#FFFF00"
set object 19 rect from 0.120443, 0 to 136.465682, 10 fc rgb "#FF00FF"
set object 20 rect from 0.121461, 0 to 137.627490, 10 fc rgb "#808080"
set object 21 rect from 0.122503, 0 to 138.237073, 10 fc rgb "#800080"
set object 22 rect from 0.123283, 0 to 138.810666, 10 fc rgb "#008080"
set object 23 rect from 0.123557, 0 to 154.405651, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.136252, 10 to 157.230862, 20 fc rgb "#FF0000"
set object 25 rect from 0.102006, 10 to 116.395548, 20 fc rgb "#00FF00"
set object 26 rect from 0.103840, 10 to 117.133339, 20 fc rgb "#0000FF"
set object 27 rect from 0.104267, 10 to 117.970110, 20 fc rgb "#FFFF00"
set object 28 rect from 0.105036, 10 to 119.315247, 20 fc rgb "#FF00FF"
set object 29 rect from 0.106229, 10 to 120.567027, 20 fc rgb "#808080"
set object 30 rect from 0.107345, 10 to 121.268837, 20 fc rgb "#800080"
set object 31 rect from 0.108207, 10 to 121.930156, 20 fc rgb "#008080"
set object 32 rect from 0.108532, 10 to 152.459930, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.134532, 20 to 154.867904, 30 fc rgb "#FF0000"
set object 34 rect from 0.088447, 20 to 100.971515, 30 fc rgb "#00FF00"
set object 35 rect from 0.090135, 20 to 101.611466, 30 fc rgb "#0000FF"
set object 36 rect from 0.090463, 20 to 102.411115, 30 fc rgb "#FFFF00"
set object 37 rect from 0.091191, 20 to 103.729262, 30 fc rgb "#FF00FF"
set object 38 rect from 0.092343, 20 to 104.760599, 30 fc rgb "#808080"
set object 39 rect from 0.093284, 20 to 105.373559, 30 fc rgb "#800080"
set object 40 rect from 0.094087, 20 to 106.059623, 30 fc rgb "#008080"
set object 41 rect from 0.094436, 20 to 150.797633, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133088, 30 to 153.387805, 40 fc rgb "#FF0000"
set object 43 rect from 0.072775, 30 to 83.455554, 40 fc rgb "#00FF00"
set object 44 rect from 0.074557, 30 to 84.159609, 40 fc rgb "#0000FF"
set object 45 rect from 0.074951, 30 to 85.072861, 40 fc rgb "#FFFF00"
set object 46 rect from 0.075757, 30 to 86.358385, 40 fc rgb "#FF00FF"
set object 47 rect from 0.076897, 30 to 87.298628, 40 fc rgb "#808080"
set object 48 rect from 0.077739, 30 to 87.961070, 40 fc rgb "#800080"
set object 49 rect from 0.078616, 30 to 88.689870, 40 fc rgb "#008080"
set object 50 rect from 0.078976, 30 to 149.149959, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.131685, 40 to 152.006686, 50 fc rgb "#FF0000"
set object 52 rect from 0.056916, 40 to 65.897979, 50 fc rgb "#00FF00"
set object 53 rect from 0.058966, 40 to 66.719005, 50 fc rgb "#0000FF"
set object 54 rect from 0.059449, 40 to 67.590636, 50 fc rgb "#FFFF00"
set object 55 rect from 0.060250, 40 to 68.970640, 50 fc rgb "#FF00FF"
set object 56 rect from 0.061463, 40 to 70.037968, 50 fc rgb "#808080"
set object 57 rect from 0.062409, 40 to 70.717287, 50 fc rgb "#800080"
set object 58 rect from 0.063264, 40 to 71.549558, 50 fc rgb "#008080"
set object 59 rect from 0.063757, 40 to 147.360576, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.129893, 50 to 152.290092, 60 fc rgb "#FF0000"
set object 61 rect from 0.036434, 50 to 43.872002, 60 fc rgb "#00FF00"
set object 62 rect from 0.039531, 50 to 45.129405, 60 fc rgb "#0000FF"
set object 63 rect from 0.040266, 50 to 47.309059, 60 fc rgb "#FFFF00"
set object 64 rect from 0.042236, 50 to 49.215412, 60 fc rgb "#FF00FF"
set object 65 rect from 0.043884, 50 to 50.410965, 60 fc rgb "#808080"
set object 66 rect from 0.044961, 50 to 51.157756, 60 fc rgb "#800080"
set object 67 rect from 0.046125, 50 to 52.611987, 60 fc rgb "#008080"
set object 68 rect from 0.046995, 50 to 145.158431, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
