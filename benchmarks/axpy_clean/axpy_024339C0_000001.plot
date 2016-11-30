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

set object 15 rect from 0.270141, 0 to 162.747228, 10 fc rgb "#FF0000"
set object 16 rect from 0.220964, 0 to 131.856400, 10 fc rgb "#00FF00"
set object 17 rect from 0.226050, 0 to 132.645479, 10 fc rgb "#0000FF"
set object 18 rect from 0.226858, 0 to 133.644132, 10 fc rgb "#FFFF00"
set object 19 rect from 0.228606, 0 to 135.013904, 10 fc rgb "#FF00FF"
set object 20 rect from 0.230915, 0 to 136.725556, 10 fc rgb "#808080"
set object 21 rect from 0.233841, 0 to 137.433846, 10 fc rgb "#800080"
set object 22 rect from 0.235639, 0 to 138.158551, 10 fc rgb "#008080"
set object 23 rect from 0.236286, 0 to 157.340120, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.266405, 10 to 161.132171, 20 fc rgb "#FF0000"
set object 25 rect from 0.187111, 10 to 112.002218, 20 fc rgb "#00FF00"
set object 26 rect from 0.192335, 10 to 113.010814, 20 fc rgb "#0000FF"
set object 27 rect from 0.193357, 10 to 114.118938, 20 fc rgb "#FFFF00"
set object 28 rect from 0.195305, 10 to 115.752138, 20 fc rgb "#FF00FF"
set object 29 rect from 0.198072, 10 to 117.526995, 20 fc rgb "#808080"
set object 30 rect from 0.201111, 10 to 118.393341, 20 fc rgb "#800080"
set object 31 rect from 0.203102, 10 to 119.173645, 20 fc rgb "#008080"
set object 32 rect from 0.203860, 10 to 155.074136, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.262339, 20 to 158.394407, 30 fc rgb "#FF0000"
set object 34 rect from 0.155710, 20 to 93.579285, 30 fc rgb "#00FF00"
set object 35 rect from 0.160711, 20 to 94.389439, 30 fc rgb "#0000FF"
set object 36 rect from 0.161504, 20 to 95.488787, 30 fc rgb "#FFFF00"
set object 37 rect from 0.163387, 20 to 96.983836, 30 fc rgb "#FF00FF"
set object 38 rect from 0.165976, 20 to 98.624048, 30 fc rgb "#808080"
set object 39 rect from 0.168751, 20 to 99.393816, 30 fc rgb "#800080"
set object 40 rect from 0.170687, 20 to 100.394806, 30 fc rgb "#008080"
set object 41 rect from 0.171785, 20 to 152.908254, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.258693, 30 to 156.665187, 40 fc rgb "#FF0000"
set object 43 rect from 0.120109, 30 to 72.777391, 40 fc rgb "#00FF00"
set object 44 rect from 0.125192, 30 to 73.605688, 40 fc rgb "#0000FF"
set object 45 rect from 0.126059, 30 to 74.939173, 40 fc rgb "#FFFF00"
set object 46 rect from 0.128288, 30 to 76.565935, 40 fc rgb "#FF00FF"
set object 47 rect from 0.131071, 30 to 78.186260, 40 fc rgb "#808080"
set object 48 rect from 0.133836, 30 to 79.026263, 40 fc rgb "#800080"
set object 49 rect from 0.135845, 30 to 79.868621, 40 fc rgb "#008080"
set object 50 rect from 0.136740, 30 to 150.687924, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.255113, 40 to 154.493461, 50 fc rgb "#FF0000"
set object 52 rect from 0.087084, 40 to 53.755030, 50 fc rgb "#00FF00"
set object 53 rect from 0.092708, 40 to 54.649480, 50 fc rgb "#0000FF"
set object 54 rect from 0.093620, 40 to 55.861213, 50 fc rgb "#FFFF00"
set object 55 rect from 0.095694, 40 to 57.468070, 50 fc rgb "#FF00FF"
set object 56 rect from 0.098473, 40 to 59.099507, 50 fc rgb "#808080"
set object 57 rect from 0.101243, 40 to 59.926636, 50 fc rgb "#800080"
set object 58 rect from 0.103301, 40 to 60.890746, 50 fc rgb "#008080"
set object 59 rect from 0.104375, 40 to 148.481639, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.250933, 50 to 155.054213, 60 fc rgb "#FF0000"
set object 61 rect from 0.042851, 50 to 28.768874, 60 fc rgb "#00FF00"
set object 62 rect from 0.050252, 50 to 30.602854, 60 fc rgb "#0000FF"
set object 63 rect from 0.052600, 50 to 32.919169, 60 fc rgb "#FFFF00"
set object 64 rect from 0.056652, 50 to 35.211514, 60 fc rgb "#FF00FF"
set object 65 rect from 0.060457, 50 to 37.147341, 60 fc rgb "#808080"
set object 66 rect from 0.063839, 50 to 38.232035, 60 fc rgb "#800080"
set object 67 rect from 0.066739, 50 to 39.845923, 60 fc rgb "#008080"
set object 68 rect from 0.068383, 50 to 145.537790, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
