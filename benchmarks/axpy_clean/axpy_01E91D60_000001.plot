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

set object 15 rect from 0.221725, 0 to 159.417835, 10 fc rgb "#FF0000"
set object 16 rect from 0.191785, 0 to 132.045225, 10 fc rgb "#00FF00"
set object 17 rect from 0.193768, 0 to 132.582575, 10 fc rgb "#0000FF"
set object 18 rect from 0.194301, 0 to 133.038674, 10 fc rgb "#FFFF00"
set object 19 rect from 0.194971, 0 to 133.773342, 10 fc rgb "#FF00FF"
set object 20 rect from 0.196057, 0 to 140.180538, 10 fc rgb "#808080"
set object 21 rect from 0.205453, 0 to 140.600447, 10 fc rgb "#800080"
set object 22 rect from 0.206327, 0 to 141.012156, 10 fc rgb "#008080"
set object 23 rect from 0.206653, 0 to 150.985497, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.218507, 10 to 157.169431, 20 fc rgb "#FF0000"
set object 25 rect from 0.138163, 10 to 95.470844, 20 fc rgb "#00FF00"
set object 26 rect from 0.140194, 10 to 95.965177, 20 fc rgb "#0000FF"
set object 27 rect from 0.140671, 10 to 96.426048, 20 fc rgb "#FFFF00"
set object 28 rect from 0.141389, 10 to 97.268603, 20 fc rgb "#FF00FF"
set object 29 rect from 0.142617, 10 to 103.572012, 20 fc rgb "#808080"
set object 30 rect from 0.151841, 10 to 104.018547, 20 fc rgb "#800080"
set object 31 rect from 0.152743, 10 to 104.459618, 20 fc rgb "#008080"
set object 32 rect from 0.153117, 10 to 148.722778, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.213195, 20 to 161.081742, 30 fc rgb "#FF0000"
set object 34 rect from 0.034443, 20 to 25.960595, 30 fc rgb "#00FF00"
set object 35 rect from 0.038484, 20 to 27.068066, 30 fc rgb "#0000FF"
set object 36 rect from 0.039775, 20 to 30.217039, 30 fc rgb "#FFFF00"
set object 37 rect from 0.044457, 20 to 35.218386, 30 fc rgb "#FF00FF"
set object 38 rect from 0.051709, 20 to 41.532040, 30 fc rgb "#808080"
set object 39 rect from 0.060969, 20 to 42.119223, 30 fc rgb "#800080"
set object 40 rect from 0.062288, 20 to 42.963142, 30 fc rgb "#008080"
set object 41 rect from 0.063069, 20 to 144.720322, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.220119, 30 to 162.623468, 40 fc rgb "#FF0000"
set object 43 rect from 0.161515, 30 to 111.307204, 40 fc rgb "#00FF00"
set object 44 rect from 0.163395, 30 to 111.804946, 40 fc rgb "#0000FF"
set object 45 rect from 0.163870, 30 to 113.915419, 40 fc rgb "#FFFF00"
set object 46 rect from 0.166962, 30 to 117.538926, 40 fc rgb "#FF00FF"
set object 47 rect from 0.172284, 30 to 123.653197, 40 fc rgb "#808080"
set object 48 rect from 0.181249, 30 to 124.168703, 40 fc rgb "#800080"
set object 49 rect from 0.182263, 30 to 124.638456, 40 fc rgb "#008080"
set object 50 rect from 0.182677, 30 to 149.887590, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.215136, 40 to 159.927857, 50 fc rgb "#FF0000"
set object 52 rect from 0.074763, 40 to 52.364317, 50 fc rgb "#00FF00"
set object 53 rect from 0.077107, 40 to 52.873668, 50 fc rgb "#0000FF"
set object 54 rect from 0.077562, 40 to 55.227202, 50 fc rgb "#FFFF00"
set object 55 rect from 0.081043, 40 to 59.312953, 50 fc rgb "#FF00FF"
set object 56 rect from 0.087013, 40 to 65.464787, 50 fc rgb "#808080"
set object 57 rect from 0.096038, 40 to 65.998718, 50 fc rgb "#800080"
set object 58 rect from 0.097129, 40 to 66.553812, 50 fc rgb "#008080"
set object 59 rect from 0.097619, 40 to 146.446394, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.216797, 50 to 160.573074, 60 fc rgb "#FF0000"
set object 61 rect from 0.107457, 50 to 74.491115, 60 fc rgb "#00FF00"
set object 62 rect from 0.109485, 50 to 74.949259, 60 fc rgb "#0000FF"
set object 63 rect from 0.109910, 50 to 77.232469, 60 fc rgb "#FFFF00"
set object 64 rect from 0.113236, 50 to 80.989116, 60 fc rgb "#FF00FF"
set object 65 rect from 0.118740, 50 to 87.097251, 60 fc rgb "#808080"
set object 66 rect from 0.127701, 50 to 87.576558, 60 fc rgb "#800080"
set object 67 rect from 0.128693, 50 to 88.038802, 60 fc rgb "#008080"
set object 68 rect from 0.129067, 50 to 147.578436, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
