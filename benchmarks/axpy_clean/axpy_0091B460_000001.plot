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

set object 15 rect from 0.251094, 0 to 166.288555, 10 fc rgb "#FF0000"
set object 16 rect from 0.206523, 0 to 124.037223, 10 fc rgb "#00FF00"
set object 17 rect from 0.208076, 0 to 124.329736, 10 fc rgb "#0000FF"
set object 18 rect from 0.208363, 0 to 124.673595, 10 fc rgb "#FFFF00"
set object 19 rect from 0.208972, 0 to 132.993604, 10 fc rgb "#FF00FF"
set object 20 rect from 0.222882, 0 to 140.506507, 10 fc rgb "#808080"
set object 21 rect from 0.235471, 0 to 140.806190, 10 fc rgb "#800080"
set object 22 rect from 0.236204, 0 to 141.077221, 10 fc rgb "#008080"
set object 23 rect from 0.236423, 0 to 149.610946, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.249776, 10 to 157.705905, 20 fc rgb "#FF0000"
set object 25 rect from 0.181233, 10 to 108.989330, 20 fc rgb "#00FF00"
set object 26 rect from 0.182866, 10 to 109.314677, 20 fc rgb "#0000FF"
set object 27 rect from 0.183212, 10 to 109.685996, 20 fc rgb "#FFFF00"
set object 28 rect from 0.183877, 10 to 110.278194, 20 fc rgb "#FF00FF"
set object 29 rect from 0.184850, 10 to 117.684839, 20 fc rgb "#808080"
set object 30 rect from 0.197287, 10 to 118.007206, 20 fc rgb "#800080"
set object 31 rect from 0.198013, 10 to 118.304496, 20 fc rgb "#008080"
set object 32 rect from 0.198286, 10 to 148.806224, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.245137, 20 to 163.691746, 30 fc rgb "#FF0000"
set object 34 rect from 0.076027, 20 to 46.300613, 30 fc rgb "#00FF00"
set object 35 rect from 0.077859, 20 to 46.714321, 30 fc rgb "#0000FF"
set object 36 rect from 0.078379, 20 to 48.351216, 30 fc rgb "#FFFF00"
set object 37 rect from 0.081122, 20 to 56.439005, 30 fc rgb "#FF00FF"
set object 38 rect from 0.094658, 20 to 63.575215, 30 fc rgb "#808080"
set object 39 rect from 0.106599, 20 to 64.052792, 30 fc rgb "#800080"
set object 40 rect from 0.107662, 20 to 64.419335, 30 fc rgb "#008080"
set object 41 rect from 0.108033, 20 to 146.014184, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.243365, 30 to 160.905675, 40 fc rgb "#FF0000"
set object 43 rect from 0.035135, 30 to 23.367902, 40 fc rgb "#00FF00"
set object 44 rect from 0.039520, 30 to 24.026362, 40 fc rgb "#0000FF"
set object 45 rect from 0.040356, 30 to 26.357540, 40 fc rgb "#FFFF00"
set object 46 rect from 0.044286, 30 to 31.651500, 40 fc rgb "#FF00FF"
set object 47 rect from 0.053128, 30 to 38.923822, 40 fc rgb "#808080"
set object 48 rect from 0.065361, 30 to 39.453937, 40 fc rgb "#800080"
set object 49 rect from 0.066599, 30 to 40.153584, 40 fc rgb "#008080"
set object 50 rect from 0.067554, 30 to 144.730693, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.246809, 40 to 161.302071, 50 fc rgb "#FF0000"
set object 52 rect from 0.115028, 40 to 69.461968, 50 fc rgb "#00FF00"
set object 53 rect from 0.116656, 40 to 69.824330, 50 fc rgb "#0000FF"
set object 54 rect from 0.117078, 40 to 71.369894, 50 fc rgb "#FFFF00"
set object 55 rect from 0.119657, 40 to 76.234029, 50 fc rgb "#FF00FF"
set object 56 rect from 0.127799, 40 to 83.344566, 50 fc rgb "#808080"
set object 57 rect from 0.139710, 40 to 83.792299, 50 fc rgb "#800080"
set object 58 rect from 0.140696, 40 to 84.100932, 50 fc rgb "#008080"
set object 59 rect from 0.140979, 40 to 146.914417, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.248389, 50 to 161.878132, 60 fc rgb "#FF0000"
set object 61 rect from 0.148317, 50 to 89.287433, 60 fc rgb "#00FF00"
set object 62 rect from 0.149866, 50 to 89.606216, 60 fc rgb "#0000FF"
set object 63 rect from 0.150199, 50 to 91.162526, 60 fc rgb "#FFFF00"
set object 64 rect from 0.152807, 50 to 95.722805, 60 fc rgb "#FF00FF"
set object 65 rect from 0.160445, 50 to 102.795136, 60 fc rgb "#808080"
set object 66 rect from 0.172294, 50 to 103.237495, 60 fc rgb "#800080"
set object 67 rect from 0.173279, 50 to 103.560459, 60 fc rgb "#008080"
set object 68 rect from 0.173575, 50 to 147.977031, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
