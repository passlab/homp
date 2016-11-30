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

set object 15 rect from 0.288356, 0 to 157.723993, 10 fc rgb "#FF0000"
set object 16 rect from 0.164416, 0 to 85.240110, 10 fc rgb "#00FF00"
set object 17 rect from 0.166270, 0 to 85.544726, 10 fc rgb "#0000FF"
set object 18 rect from 0.166640, 0 to 85.842146, 10 fc rgb "#FFFF00"
set object 19 rect from 0.167245, 0 to 86.330140, 10 fc rgb "#FF00FF"
set object 20 rect from 0.168169, 0 to 94.922470, 10 fc rgb "#808080"
set object 21 rect from 0.184910, 0 to 95.196262, 10 fc rgb "#800080"
set object 22 rect from 0.185670, 0 to 95.438211, 10 fc rgb "#008080"
set object 23 rect from 0.185910, 0 to 147.822295, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.286658, 10 to 156.961175, 20 fc rgb "#FF0000"
set object 25 rect from 0.134309, 10 to 69.833298, 20 fc rgb "#00FF00"
set object 26 rect from 0.136272, 10 to 70.163082, 20 fc rgb "#0000FF"
set object 27 rect from 0.136696, 10 to 70.503650, 20 fc rgb "#FFFF00"
set object 28 rect from 0.137388, 10 to 71.060480, 20 fc rgb "#FF00FF"
set object 29 rect from 0.138459, 10 to 79.610175, 20 fc rgb "#808080"
set object 30 rect from 0.155097, 10 to 79.904006, 20 fc rgb "#800080"
set object 31 rect from 0.155899, 10 to 80.192693, 20 fc rgb "#008080"
set object 32 rect from 0.156230, 10 to 146.864794, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.282473, 20 to 163.210108, 30 fc rgb "#FF0000"
set object 34 rect from 0.040174, 20 to 22.159147, 30 fc rgb "#00FF00"
set object 35 rect from 0.043667, 20 to 22.786864, 30 fc rgb "#0000FF"
set object 36 rect from 0.044486, 20 to 25.029590, 30 fc rgb "#FFFF00"
set object 37 rect from 0.048893, 20 to 31.663771, 30 fc rgb "#FF00FF"
set object 38 rect from 0.061773, 20 to 40.198571, 30 fc rgb "#808080"
set object 39 rect from 0.078393, 20 to 40.768237, 30 fc rgb "#800080"
set object 40 rect from 0.079948, 20 to 41.502290, 30 fc rgb "#008080"
set object 41 rect from 0.080932, 20 to 144.575835, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.284753, 30 to 162.834605, 40 fc rgb "#FF0000"
set object 43 rect from 0.091007, 30 to 47.673646, 40 fc rgb "#00FF00"
set object 44 rect from 0.093166, 30 to 47.990072, 40 fc rgb "#0000FF"
set object 45 rect from 0.093534, 30 to 49.422729, 40 fc rgb "#FFFF00"
set object 46 rect from 0.096353, 30 to 55.656752, 40 fc rgb "#FF00FF"
set object 47 rect from 0.108480, 30 to 64.081106, 40 fc rgb "#808080"
set object 48 rect from 0.124872, 30 to 64.623553, 40 fc rgb "#800080"
set object 49 rect from 0.126204, 30 to 64.972855, 40 fc rgb "#008080"
set object 50 rect from 0.126636, 30 to 145.775798, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.291256, 40 to 165.814468, 50 fc rgb "#FF0000"
set object 52 rect from 0.241195, 40 to 124.571587, 50 fc rgb "#00FF00"
set object 53 rect from 0.242848, 40 to 124.873631, 50 fc rgb "#0000FF"
set object 54 rect from 0.243211, 40 to 126.370494, 50 fc rgb "#FFFF00"
set object 55 rect from 0.246144, 40 to 132.193573, 50 fc rgb "#FF00FF"
set object 56 rect from 0.257454, 40 to 140.636420, 50 fc rgb "#808080"
set object 57 rect from 0.273892, 40 to 141.132642, 50 fc rgb "#800080"
set object 58 rect from 0.275115, 40 to 141.455230, 50 fc rgb "#008080"
set object 59 rect from 0.275486, 40 to 149.355118, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.289800, 50 to 165.273552, 60 fc rgb "#FF0000"
set object 61 rect from 0.198177, 50 to 102.509016, 60 fc rgb "#00FF00"
set object 62 rect from 0.199882, 50 to 102.838800, 60 fc rgb "#0000FF"
set object 63 rect from 0.200315, 50 to 104.276593, 60 fc rgb "#FFFF00"
set object 64 rect from 0.203136, 50 to 110.390411, 60 fc rgb "#FF00FF"
set object 65 rect from 0.215022, 50 to 118.759293, 60 fc rgb "#808080"
set object 66 rect from 0.231315, 50 to 119.283759, 60 fc rgb "#800080"
set object 67 rect from 0.232585, 50 to 119.607894, 60 fc rgb "#008080"
set object 68 rect from 0.232962, 50 to 148.615416, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
