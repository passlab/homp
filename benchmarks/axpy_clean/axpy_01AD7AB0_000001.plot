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

set object 15 rect from 10.578029, 0 to 157.163654, 10 fc rgb "#FF0000"
set object 16 rect from 1.012870, 0 to 13.828787, 10 fc rgb "#00FF00"
set object 17 rect from 1.015608, 0 to 13.838772, 10 fc rgb "#0000FF"
set object 18 rect from 1.016082, 0 to 13.850649, 10 fc rgb "#FFFF00"
set object 19 rect from 1.016954, 0 to 13.866191, 10 fc rgb "#FF00FF"
set object 20 rect from 1.018129, 0 to 26.905060, 10 fc rgb "#808080"
set object 21 rect from 1.975496, 0 to 26.918287, 10 fc rgb "#800080"
set object 22 rect from 1.976611, 0 to 26.927509, 10 fc rgb "#008080"
set object 23 rect from 1.976991, 0 to 144.075939, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 10.575254, 10 to 157.050339, 20 fc rgb "#FF0000"
set object 25 rect from 0.034664, 10 to 0.534408, 20 fc rgb "#00FF00"
set object 26 rect from 0.039672, 10 to 0.553682, 20 fc rgb "#0000FF"
set object 27 rect from 0.040798, 10 to 0.584303, 20 fc rgb "#FFFF00"
set object 28 rect from 0.043040, 10 to 0.608168, 20 fc rgb "#FF00FF"
set object 29 rect from 0.044780, 10 to 13.533203, 20 fc rgb "#808080"
set object 30 rect from 0.993729, 10 to 13.548295, 20 fc rgb "#800080"
set object 31 rect from 0.995247, 10 to 13.560405, 20 fc rgb "#008080"
set object 32 rect from 0.995714, 10 to 144.036247, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 10.580389, 20 to 220.985904, 30 fc rgb "#FF0000"
set object 34 rect from 3.619523, 20 to 49.330510, 30 fc rgb "#00FF00"
set object 35 rect from 3.621941, 20 to 49.341788, 30 fc rgb "#0000FF"
set object 36 rect from 3.622486, 20 to 49.421896, 30 fc rgb "#FFFF00"
set object 37 rect from 3.628427, 20 to 104.651610, 30 fc rgb "#FF00FF"
set object 38 rect from 7.683002, 20 to 125.184464, 30 fc rgb "#808080"
set object 39 rect from 9.190761, 20 to 126.212620, 30 fc rgb "#800080"
set object 40 rect from 9.266409, 20 to 126.234046, 30 fc rgb "#008080"
set object 41 rect from 9.267519, 20 to 144.109625, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 10.586206, 30 to 160.802799, 40 fc rgb "#FF0000"
set object 43 rect from 6.423204, 30 to 87.541173, 40 fc rgb "#00FF00"
set object 44 rect from 6.427225, 30 to 87.560025, 40 fc rgb "#0000FF"
set object 45 rect from 6.428271, 30 to 87.659243, 40 fc rgb "#FFFF00"
set object 46 rect from 6.435621, 30 to 95.213406, 40 fc rgb "#FF00FF"
set object 47 rect from 6.990111, 30 to 103.501025, 40 fc rgb "#808080"
set object 48 rect from 7.598559, 30 to 104.158639, 40 fc rgb "#800080"
set object 49 rect from 7.647318, 30 to 104.182667, 40 fc rgb "#008080"
set object 50 rect from 7.648576, 30 to 144.188615, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 10.583463, 40 to 165.105919, 50 fc rgb "#FF0000"
set object 52 rect from 1.990747, 40 to 27.148965, 50 fc rgb "#00FF00"
set object 53 rect from 1.993513, 40 to 27.160394, 50 fc rgb "#0000FF"
set object 54 rect from 1.994096, 40 to 27.255866, 50 fc rgb "#FFFF00"
set object 55 rect from 2.001187, 40 to 34.616796, 50 fc rgb "#FF00FF"
set object 56 rect from 2.541511, 40 to 47.612336, 50 fc rgb "#808080"
set object 57 rect from 3.495718, 40 to 48.108849, 50 fc rgb "#800080"
set object 58 rect from 3.532326, 40 to 49.134757, 50 fc rgb "#008080"
set object 59 rect from 3.607474, 40 to 144.148773, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 10.587705, 50 to 161.389459, 60 fc rgb "#FF0000"
set object 61 rect from 9.294672, 50 to 126.643423, 60 fc rgb "#00FF00"
set object 62 rect from 9.297755, 50 to 126.654852, 60 fc rgb "#0000FF"
set object 63 rect from 9.298347, 50 to 126.727685, 60 fc rgb "#FFFF00"
set object 64 rect from 9.303764, 50 to 134.729270, 60 fc rgb "#FF00FF"
set object 65 rect from 9.891118, 50 to 143.213869, 60 fc rgb "#808080"
set object 66 rect from 10.514160, 50 to 143.826273, 60 fc rgb "#800080"
set object 67 rect from 10.559344, 50 to 143.842414, 60 fc rgb "#008080"
set object 68 rect from 10.560167, 50 to 144.211595, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
