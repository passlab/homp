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

set object 15 rect from 0.133215, 0 to 160.574894, 10 fc rgb "#FF0000"
set object 16 rect from 0.112124, 0 to 133.186465, 10 fc rgb "#00FF00"
set object 17 rect from 0.114015, 0 to 133.893308, 10 fc rgb "#0000FF"
set object 18 rect from 0.114346, 0 to 134.629442, 10 fc rgb "#FFFF00"
set object 19 rect from 0.114972, 0 to 135.784055, 10 fc rgb "#FF00FF"
set object 20 rect from 0.115958, 0 to 137.876443, 10 fc rgb "#808080"
set object 21 rect from 0.117743, 0 to 138.495356, 10 fc rgb "#800080"
set object 22 rect from 0.118527, 0 to 139.127177, 10 fc rgb "#008080"
set object 23 rect from 0.118811, 0 to 155.499277, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.131686, 10 to 159.671127, 20 fc rgb "#FF0000"
set object 25 rect from 0.096649, 10 to 115.189694, 20 fc rgb "#00FF00"
set object 26 rect from 0.098639, 10 to 115.972711, 20 fc rgb "#0000FF"
set object 27 rect from 0.099058, 10 to 116.960879, 20 fc rgb "#FFFF00"
set object 28 rect from 0.099942, 10 to 118.373376, 20 fc rgb "#FF00FF"
set object 29 rect from 0.101136, 10 to 120.701378, 20 fc rgb "#808080"
set object 30 rect from 0.103105, 10 to 121.462124, 20 fc rgb "#800080"
set object 31 rect from 0.104000, 10 to 122.163098, 20 fc rgb "#008080"
set object 32 rect from 0.104362, 10 to 153.567499, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.129854, 20 to 158.661893, 30 fc rgb "#FF0000"
set object 34 rect from 0.080798, 20 to 96.438012, 30 fc rgb "#00FF00"
set object 35 rect from 0.082668, 20 to 97.165954, 30 fc rgb "#0000FF"
set object 36 rect from 0.083013, 20 to 99.760033, 30 fc rgb "#FFFF00"
set object 37 rect from 0.085228, 20 to 101.258102, 30 fc rgb "#FF00FF"
set object 38 rect from 0.086502, 20 to 102.971855, 30 fc rgb "#808080"
set object 39 rect from 0.087986, 20 to 103.784181, 30 fc rgb "#800080"
set object 40 rect from 0.088916, 20 to 104.476962, 30 fc rgb "#008080"
set object 41 rect from 0.089251, 20 to 151.673224, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.128299, 30 to 156.733609, 40 fc rgb "#FF0000"
set object 43 rect from 0.064351, 30 to 77.276078, 40 fc rgb "#00FF00"
set object 44 rect from 0.066295, 30 to 77.998152, 40 fc rgb "#0000FF"
set object 45 rect from 0.066659, 30 to 80.384739, 40 fc rgb "#FFFF00"
set object 46 rect from 0.068702, 30 to 81.899227, 40 fc rgb "#FF00FF"
set object 47 rect from 0.069987, 30 to 83.753642, 40 fc rgb "#808080"
set object 48 rect from 0.071570, 30 to 84.483925, 40 fc rgb "#800080"
set object 49 rect from 0.072470, 30 to 85.195449, 40 fc rgb "#008080"
set object 50 rect from 0.072801, 30 to 149.729708, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.126758, 40 to 155.147629, 50 fc rgb "#FF0000"
set object 52 rect from 0.047728, 40 to 58.192660, 50 fc rgb "#00FF00"
set object 53 rect from 0.050024, 40 to 58.944043, 50 fc rgb "#0000FF"
set object 54 rect from 0.050408, 40 to 61.303678, 50 fc rgb "#FFFF00"
set object 55 rect from 0.052423, 40 to 63.043230, 50 fc rgb "#FF00FF"
set object 56 rect from 0.053929, 40 to 64.914065, 50 fc rgb "#808080"
set object 57 rect from 0.055531, 40 to 65.705291, 50 fc rgb "#800080"
set object 58 rect from 0.056482, 40 to 66.586787, 50 fc rgb "#008080"
set object 59 rect from 0.056940, 40 to 147.827241, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.124992, 50 to 155.539120, 60 fc rgb "#FF0000"
set object 61 rect from 0.025938, 50 to 33.723006, 60 fc rgb "#00FF00"
set object 62 rect from 0.029240, 50 to 34.869409, 60 fc rgb "#0000FF"
set object 63 rect from 0.029879, 50 to 38.548948, 60 fc rgb "#FFFF00"
set object 64 rect from 0.033059, 50 to 40.804252, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034972, 50 to 42.975172, 60 fc rgb "#808080"
set object 66 rect from 0.036816, 50 to 43.885978, 60 fc rgb "#800080"
set object 67 rect from 0.037977, 50 to 45.304327, 60 fc rgb "#008080"
set object 68 rect from 0.038788, 50 to 145.404305, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
