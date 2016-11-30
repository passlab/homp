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

set object 15 rect from 0.142935, 0 to 154.067673, 10 fc rgb "#FF0000"
set object 16 rect from 0.075584, 0 to 81.394658, 10 fc rgb "#00FF00"
set object 17 rect from 0.077907, 0 to 82.268911, 10 fc rgb "#0000FF"
set object 18 rect from 0.078406, 0 to 83.068569, 10 fc rgb "#FFFF00"
set object 19 rect from 0.079173, 0 to 84.157181, 10 fc rgb "#FF00FF"
set object 20 rect from 0.080218, 0 to 85.550531, 10 fc rgb "#808080"
set object 21 rect from 0.081534, 0 to 86.121111, 10 fc rgb "#800080"
set object 22 rect from 0.082348, 0 to 86.700115, 10 fc rgb "#008080"
set object 23 rect from 0.082645, 0 to 149.481986, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.141114, 10 to 152.656474, 20 fc rgb "#FF0000"
set object 25 rect from 0.059112, 10 to 64.488382, 20 fc rgb "#00FF00"
set object 26 rect from 0.061755, 10 to 65.421510, 20 fc rgb "#0000FF"
set object 27 rect from 0.062374, 10 to 66.311515, 20 fc rgb "#FFFF00"
set object 28 rect from 0.063261, 10 to 67.626073, 20 fc rgb "#FF00FF"
set object 29 rect from 0.064523, 10 to 69.104540, 20 fc rgb "#808080"
set object 30 rect from 0.065937, 10 to 69.788610, 20 fc rgb "#800080"
set object 31 rect from 0.066843, 10 to 70.449569, 20 fc rgb "#008080"
set object 32 rect from 0.067163, 10 to 147.546429, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.139188, 20 to 154.832727, 30 fc rgb "#FF0000"
set object 34 rect from 0.035317, 20 to 40.386281, 30 fc rgb "#00FF00"
set object 35 rect from 0.038925, 20 to 42.599269, 30 fc rgb "#0000FF"
set object 36 rect from 0.040687, 20 to 45.066543, 30 fc rgb "#FFFF00"
set object 37 rect from 0.043075, 20 to 47.312099, 30 fc rgb "#FF00FF"
set object 38 rect from 0.045163, 20 to 49.119448, 30 fc rgb "#808080"
set object 39 rect from 0.046890, 20 to 49.988471, 30 fc rgb "#800080"
set object 40 rect from 0.048247, 20 to 51.317717, 30 fc rgb "#008080"
set object 41 rect from 0.048975, 20 to 145.173729, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.144766, 30 to 156.819737, 40 fc rgb "#FF0000"
set object 43 rect from 0.092234, 30 to 98.925064, 40 fc rgb "#00FF00"
set object 44 rect from 0.094508, 30 to 99.829788, 40 fc rgb "#0000FF"
set object 45 rect from 0.095119, 30 to 100.944706, 40 fc rgb "#FFFF00"
set object 46 rect from 0.096196, 30 to 102.438925, 40 fc rgb "#FF00FF"
set object 47 rect from 0.097617, 30 to 103.700934, 40 fc rgb "#808080"
set object 48 rect from 0.098818, 30 to 104.421769, 40 fc rgb "#800080"
set object 49 rect from 0.099747, 30 to 105.166781, 40 fc rgb "#008080"
set object 50 rect from 0.100201, 30 to 151.479550, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.147787, 40 to 159.385751, 50 fc rgb "#FF0000"
set object 52 rect from 0.124929, 40 to 133.094839, 50 fc rgb "#00FF00"
set object 53 rect from 0.127011, 40 to 133.837752, 50 fc rgb "#0000FF"
set object 54 rect from 0.127483, 40 to 134.736181, 50 fc rgb "#FFFF00"
set object 55 rect from 0.128343, 40 to 135.984537, 50 fc rgb "#FF00FF"
set object 56 rect from 0.129530, 40 to 137.265461, 50 fc rgb "#808080"
set object 57 rect from 0.130749, 40 to 137.904342, 50 fc rgb "#800080"
set object 58 rect from 0.131610, 40 to 138.577890, 50 fc rgb "#008080"
set object 59 rect from 0.132011, 40 to 154.808488, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.146292, 50 to 157.969321, 60 fc rgb "#FF0000"
set object 61 rect from 0.108589, 50 to 115.905935, 60 fc rgb "#00FF00"
set object 62 rect from 0.110762, 50 to 116.775992, 60 fc rgb "#0000FF"
set object 63 rect from 0.111244, 50 to 117.796303, 60 fc rgb "#FFFF00"
set object 64 rect from 0.112214, 50 to 119.113993, 60 fc rgb "#FF00FF"
set object 65 rect from 0.113486, 50 to 120.356054, 60 fc rgb "#808080"
set object 66 rect from 0.114657, 50 to 121.011752, 60 fc rgb "#800080"
set object 67 rect from 0.115531, 50 to 121.724194, 60 fc rgb "#008080"
set object 68 rect from 0.115957, 50 to 153.181865, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
