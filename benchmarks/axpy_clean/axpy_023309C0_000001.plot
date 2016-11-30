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

set object 15 rect from 0.167962, 0 to 153.493780, 10 fc rgb "#FF0000"
set object 16 rect from 0.071346, 0 to 65.753284, 10 fc rgb "#00FF00"
set object 17 rect from 0.074206, 0 to 66.434592, 10 fc rgb "#0000FF"
set object 18 rect from 0.074731, 0 to 67.253946, 10 fc rgb "#FFFF00"
set object 19 rect from 0.075662, 0 to 68.351169, 10 fc rgb "#FF00FF"
set object 20 rect from 0.076891, 0 to 69.817114, 10 fc rgb "#808080"
set object 21 rect from 0.078550, 0 to 70.411152, 10 fc rgb "#800080"
set object 22 rect from 0.079486, 0 to 70.994493, 10 fc rgb "#008080"
set object 23 rect from 0.079850, 0 to 148.580365, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.176054, 10 to 160.434320, 20 fc rgb "#FF0000"
set object 25 rect from 0.127869, 10 to 115.676106, 20 fc rgb "#00FF00"
set object 26 rect from 0.130267, 10 to 116.412622, 20 fc rgb "#0000FF"
set object 27 rect from 0.130826, 10 to 117.299685, 20 fc rgb "#FFFF00"
set object 28 rect from 0.131857, 10 to 118.289147, 20 fc rgb "#FF00FF"
set object 29 rect from 0.132950, 10 to 119.503022, 20 fc rgb "#808080"
set object 30 rect from 0.134314, 10 to 120.008887, 20 fc rgb "#800080"
set object 31 rect from 0.135260, 10 to 120.621637, 20 fc rgb "#008080"
set object 32 rect from 0.135559, 10 to 156.175432, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.173133, 20 to 158.310237, 30 fc rgb "#FF0000"
set object 34 rect from 0.110847, 20 to 100.444992, 30 fc rgb "#00FF00"
set object 35 rect from 0.113153, 20 to 101.163698, 30 fc rgb "#0000FF"
set object 36 rect from 0.113716, 20 to 101.980398, 30 fc rgb "#FFFF00"
set object 37 rect from 0.114620, 20 to 103.385774, 30 fc rgb "#FF00FF"
set object 38 rect from 0.116204, 20 to 104.480316, 30 fc rgb "#808080"
set object 39 rect from 0.117431, 20 to 104.998656, 30 fc rgb "#800080"
set object 40 rect from 0.118265, 20 to 105.519650, 30 fc rgb "#008080"
set object 41 rect from 0.118604, 20 to 153.379835, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.170770, 30 to 155.979472, 40 fc rgb "#FF0000"
set object 43 rect from 0.091179, 30 to 83.139668, 40 fc rgb "#00FF00"
set object 44 rect from 0.093724, 30 to 83.824533, 40 fc rgb "#0000FF"
set object 45 rect from 0.094236, 30 to 84.618963, 40 fc rgb "#FFFF00"
set object 46 rect from 0.095148, 30 to 85.985137, 40 fc rgb "#FF00FF"
set object 47 rect from 0.096692, 30 to 87.112644, 40 fc rgb "#808080"
set object 48 rect from 0.097942, 30 to 87.725394, 40 fc rgb "#800080"
set object 49 rect from 0.098923, 30 to 88.386212, 40 fc rgb "#008080"
set object 50 rect from 0.099362, 30 to 151.232582, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.177836, 40 to 162.140656, 50 fc rgb "#FF0000"
set object 52 rect from 0.147476, 40 to 133.048237, 50 fc rgb "#00FF00"
set object 53 rect from 0.149739, 40 to 133.748231, 50 fc rgb "#0000FF"
set object 54 rect from 0.150291, 40 to 134.579184, 50 fc rgb "#FFFF00"
set object 55 rect from 0.151236, 40 to 135.832261, 50 fc rgb "#FF00FF"
set object 56 rect from 0.152632, 40 to 136.884947, 50 fc rgb "#808080"
set object 57 rect from 0.153830, 40 to 137.450478, 50 fc rgb "#800080"
set object 58 rect from 0.154707, 40 to 137.979515, 50 fc rgb "#008080"
set object 59 rect from 0.155043, 40 to 157.783855, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.164900, 50 to 154.587446, 60 fc rgb "#FF0000"
set object 61 rect from 0.042879, 50 to 41.841452, 60 fc rgb "#00FF00"
set object 62 rect from 0.047442, 50 to 43.116799, 60 fc rgb "#0000FF"
set object 63 rect from 0.048559, 50 to 46.416482, 60 fc rgb "#FFFF00"
set object 64 rect from 0.052286, 50 to 48.276073, 60 fc rgb "#FF00FF"
set object 65 rect from 0.054336, 50 to 49.679671, 60 fc rgb "#808080"
set object 66 rect from 0.055925, 50 to 50.385902, 60 fc rgb "#800080"
set object 67 rect from 0.057203, 50 to 51.882132, 60 fc rgb "#008080"
set object 68 rect from 0.058477, 50 to 145.468599, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
