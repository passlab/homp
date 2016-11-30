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

set object 15 rect from 0.166619, 0 to 153.302331, 10 fc rgb "#FF0000"
set object 16 rect from 0.096433, 0 to 88.461189, 10 fc rgb "#00FF00"
set object 17 rect from 0.098815, 0 to 89.297536, 10 fc rgb "#0000FF"
set object 18 rect from 0.099412, 0 to 89.989994, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100197, 0 to 90.943251, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101253, 0 to 92.165395, 10 fc rgb "#808080"
set object 21 rect from 0.102619, 0 to 92.664505, 10 fc rgb "#800080"
set object 22 rect from 0.103426, 0 to 93.168112, 10 fc rgb "#008080"
set object 23 rect from 0.103723, 0 to 149.250098, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.164904, 10 to 152.556814, 20 fc rgb "#FF0000"
set object 25 rect from 0.063318, 10 to 70.559774, 20 fc rgb "#00FF00"
set object 26 rect from 0.078904, 10 to 71.548103, 20 fc rgb "#0000FF"
set object 27 rect from 0.079691, 10 to 72.389845, 20 fc rgb "#FFFF00"
set object 28 rect from 0.080652, 10 to 73.684833, 20 fc rgb "#FF00FF"
set object 29 rect from 0.082093, 10 to 75.079644, 20 fc rgb "#808080"
set object 30 rect from 0.083647, 10 to 75.647100, 20 fc rgb "#800080"
set object 31 rect from 0.084588, 10 to 76.273910, 20 fc rgb "#008080"
set object 32 rect from 0.084946, 10 to 147.697011, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.162767, 20 to 153.846405, 30 fc rgb "#FF0000"
set object 34 rect from 0.038094, 20 to 37.471025, 30 fc rgb "#00FF00"
set object 35 rect from 0.042210, 20 to 39.154509, 30 fc rgb "#0000FF"
set object 36 rect from 0.043662, 20 to 41.346098, 30 fc rgb "#FFFF00"
set object 37 rect from 0.046199, 20 to 43.518801, 30 fc rgb "#FF00FF"
set object 38 rect from 0.048525, 20 to 44.935195, 30 fc rgb "#808080"
set object 39 rect from 0.050107, 20 to 45.870463, 30 fc rgb "#800080"
set object 40 rect from 0.051816, 20 to 47.182538, 30 fc rgb "#008080"
set object 41 rect from 0.052622, 20 to 145.328262, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.168554, 30 to 155.373415, 40 fc rgb "#FF0000"
set object 43 rect from 0.114416, 30 to 104.760955, 40 fc rgb "#00FF00"
set object 44 rect from 0.116865, 30 to 105.473199, 40 fc rgb "#0000FF"
set object 45 rect from 0.117398, 30 to 106.283466, 40 fc rgb "#FFFF00"
set object 46 rect from 0.118314, 30 to 107.516403, 40 fc rgb "#FF00FF"
set object 47 rect from 0.119695, 30 to 108.648618, 40 fc rgb "#808080"
set object 48 rect from 0.120941, 30 to 109.234960, 40 fc rgb "#800080"
set object 49 rect from 0.121866, 30 to 110.025441, 40 fc rgb "#008080"
set object 50 rect from 0.122462, 30 to 150.821171, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.171704, 40 to 157.917525, 50 fc rgb "#FF0000"
set object 52 rect from 0.147241, 40 to 134.001611, 50 fc rgb "#00FF00"
set object 53 rect from 0.149384, 40 to 134.669789, 50 fc rgb "#0000FF"
set object 54 rect from 0.149884, 40 to 135.403616, 50 fc rgb "#FFFF00"
set object 55 rect from 0.150680, 40 to 136.529536, 50 fc rgb "#FF00FF"
set object 56 rect from 0.151937, 40 to 137.629376, 50 fc rgb "#808080"
set object 57 rect from 0.153166, 40 to 138.186941, 50 fc rgb "#800080"
set object 58 rect from 0.154060, 40 to 138.840730, 50 fc rgb "#008080"
set object 59 rect from 0.154509, 40 to 153.996590, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.170164, 50 to 156.537106, 60 fc rgb "#FF0000"
set object 61 rect from 0.130956, 50 to 119.254033, 60 fc rgb "#00FF00"
set object 62 rect from 0.133047, 50 to 119.913217, 60 fc rgb "#0000FF"
set object 63 rect from 0.133454, 50 to 120.739672, 60 fc rgb "#FFFF00"
set object 64 rect from 0.134386, 50 to 121.843109, 60 fc rgb "#FF00FF"
set object 65 rect from 0.135602, 50 to 122.911475, 60 fc rgb "#808080"
set object 66 rect from 0.136799, 50 to 123.483429, 60 fc rgb "#800080"
set object 67 rect from 0.137720, 50 to 124.085957, 60 fc rgb "#008080"
set object 68 rect from 0.138100, 50 to 152.561311, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
