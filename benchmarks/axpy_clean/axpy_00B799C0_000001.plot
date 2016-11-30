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

set object 15 rect from 0.153825, 0 to 159.045855, 10 fc rgb "#FF0000"
set object 16 rect from 0.129682, 0 to 132.868995, 10 fc rgb "#00FF00"
set object 17 rect from 0.132044, 0 to 133.714528, 10 fc rgb "#0000FF"
set object 18 rect from 0.132638, 0 to 134.471235, 10 fc rgb "#FFFF00"
set object 19 rect from 0.133392, 0 to 135.526626, 10 fc rgb "#FF00FF"
set object 20 rect from 0.134443, 0 to 136.920031, 10 fc rgb "#808080"
set object 21 rect from 0.135830, 0 to 137.439638, 10 fc rgb "#800080"
set object 22 rect from 0.136591, 0 to 138.001643, 10 fc rgb "#008080"
set object 23 rect from 0.136890, 0 to 154.704224, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.152281, 10 to 157.544469, 20 fc rgb "#FF0000"
set object 25 rect from 0.098494, 10 to 101.356668, 20 fc rgb "#00FF00"
set object 26 rect from 0.100805, 10 to 102.068991, 20 fc rgb "#0000FF"
set object 27 rect from 0.101273, 10 to 102.869119, 20 fc rgb "#FFFF00"
set object 28 rect from 0.102105, 10 to 104.101080, 20 fc rgb "#FF00FF"
set object 29 rect from 0.103326, 10 to 105.430917, 20 fc rgb "#808080"
set object 30 rect from 0.104637, 10 to 106.056460, 20 fc rgb "#800080"
set object 31 rect from 0.105498, 10 to 106.661855, 20 fc rgb "#008080"
set object 32 rect from 0.105853, 10 to 152.912303, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.150298, 20 to 155.512440, 30 fc rgb "#FF0000"
set object 34 rect from 0.083608, 20 to 86.196818, 30 fc rgb "#00FF00"
set object 35 rect from 0.085781, 20 to 86.840523, 30 fc rgb "#0000FF"
set object 36 rect from 0.086204, 20 to 87.727432, 30 fc rgb "#FFFF00"
set object 37 rect from 0.087061, 20 to 88.960385, 30 fc rgb "#FF00FF"
set object 38 rect from 0.088283, 20 to 90.148955, 30 fc rgb "#808080"
set object 39 rect from 0.089468, 20 to 90.784631, 30 fc rgb "#800080"
set object 40 rect from 0.090398, 20 to 91.423283, 30 fc rgb "#008080"
set object 41 rect from 0.090730, 20 to 151.132471, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.148494, 30 to 153.560067, 40 fc rgb "#FF0000"
set object 43 rect from 0.065405, 30 to 68.024154, 40 fc rgb "#00FF00"
set object 44 rect from 0.067784, 30 to 68.623475, 40 fc rgb "#0000FF"
set object 45 rect from 0.068145, 30 to 69.539612, 40 fc rgb "#FFFF00"
set object 46 rect from 0.069038, 30 to 70.689843, 40 fc rgb "#FF00FF"
set object 47 rect from 0.070185, 30 to 71.892546, 40 fc rgb "#808080"
set object 48 rect from 0.071374, 30 to 72.507022, 40 fc rgb "#800080"
set object 49 rect from 0.072245, 30 to 73.198177, 40 fc rgb "#008080"
set object 50 rect from 0.072684, 30 to 149.269886, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.146787, 40 to 152.043556, 50 fc rgb "#FF0000"
set object 52 rect from 0.048780, 40 to 51.361926, 50 fc rgb "#00FF00"
set object 53 rect from 0.051278, 40 to 52.046014, 50 fc rgb "#0000FF"
set object 54 rect from 0.051717, 40 to 52.931871, 50 fc rgb "#FFFF00"
set object 55 rect from 0.052578, 40 to 54.195134, 50 fc rgb "#FF00FF"
set object 56 rect from 0.053850, 40 to 55.503773, 50 fc rgb "#808080"
set object 57 rect from 0.055149, 40 to 56.077866, 50 fc rgb "#800080"
set object 58 rect from 0.055959, 40 to 56.737748, 50 fc rgb "#008080"
set object 59 rect from 0.056371, 40 to 147.479980, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.144902, 50 to 153.481375, 60 fc rgb "#FF0000"
set object 61 rect from 0.025559, 50 to 28.552014, 60 fc rgb "#00FF00"
set object 62 rect from 0.028717, 50 to 29.695178, 60 fc rgb "#0000FF"
set object 63 rect from 0.029560, 50 to 32.477930, 60 fc rgb "#FFFF00"
set object 64 rect from 0.032363, 50 to 34.325359, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034144, 50 to 35.868091, 60 fc rgb "#808080"
set object 66 rect from 0.035705, 50 to 36.707550, 60 fc rgb "#800080"
set object 67 rect from 0.036954, 50 to 38.003078, 60 fc rgb "#008080"
set object 68 rect from 0.037813, 50 to 145.249130, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
