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

set object 15 rect from 0.156264, 0 to 159.339727, 10 fc rgb "#FF0000"
set object 16 rect from 0.113043, 0 to 111.870073, 10 fc rgb "#00FF00"
set object 17 rect from 0.114986, 0 to 112.492831, 10 fc rgb "#0000FF"
set object 18 rect from 0.115364, 0 to 113.148795, 10 fc rgb "#FFFF00"
set object 19 rect from 0.116080, 0 to 114.170792, 10 fc rgb "#FF00FF"
set object 20 rect from 0.117089, 0 to 118.944983, 10 fc rgb "#808080"
set object 21 rect from 0.121979, 0 to 119.448658, 10 fc rgb "#800080"
set object 22 rect from 0.122754, 0 to 119.960143, 10 fc rgb "#008080"
set object 23 rect from 0.123036, 0 to 151.995395, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.154649, 10 to 158.132249, 20 fc rgb "#FF0000"
set object 25 rect from 0.095235, 10 to 94.664023, 20 fc rgb "#00FF00"
set object 26 rect from 0.097370, 10 to 95.394154, 20 fc rgb "#0000FF"
set object 27 rect from 0.097862, 10 to 96.128197, 20 fc rgb "#FFFF00"
set object 28 rect from 0.098634, 10 to 97.312229, 20 fc rgb "#FF00FF"
set object 29 rect from 0.099831, 10 to 102.062028, 20 fc rgb "#808080"
set object 30 rect from 0.104711, 10 to 102.668203, 20 fc rgb "#800080"
set object 31 rect from 0.105569, 10 to 103.270466, 20 fc rgb "#008080"
set object 32 rect from 0.105942, 10 to 150.322321, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.152990, 20 to 159.615957, 30 fc rgb "#FF0000"
set object 34 rect from 0.072995, 20 to 72.851664, 30 fc rgb "#00FF00"
set object 35 rect from 0.075013, 20 to 73.642332, 30 fc rgb "#0000FF"
set object 36 rect from 0.075583, 20 to 76.905484, 30 fc rgb "#FFFF00"
set object 37 rect from 0.078930, 20 to 78.800142, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080870, 20 to 83.327383, 30 fc rgb "#808080"
set object 39 rect from 0.085502, 20 to 83.931595, 30 fc rgb "#800080"
set object 40 rect from 0.086381, 20 to 84.574861, 30 fc rgb "#008080"
set object 41 rect from 0.086792, 20 to 148.751748, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.151430, 30 to 157.972163, 40 fc rgb "#FF0000"
set object 43 rect from 0.050887, 30 to 51.606441, 40 fc rgb "#00FF00"
set object 44 rect from 0.053256, 30 to 52.250681, 40 fc rgb "#0000FF"
set object 45 rect from 0.053650, 30 to 55.348875, 40 fc rgb "#FFFF00"
set object 46 rect from 0.056846, 30 to 57.330397, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058870, 30 to 61.892779, 40 fc rgb "#808080"
set object 48 rect from 0.063532, 30 to 62.538969, 40 fc rgb "#800080"
set object 49 rect from 0.064483, 30 to 63.213464, 40 fc rgb "#008080"
set object 50 rect from 0.064903, 30 to 147.187037, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.149659, 40 to 158.563677, 50 fc rgb "#FF0000"
set object 52 rect from 0.023888, 40 to 26.055545, 50 fc rgb "#00FF00"
set object 53 rect from 0.027266, 40 to 27.199563, 50 fc rgb "#0000FF"
set object 54 rect from 0.028002, 40 to 31.650659, 50 fc rgb "#FFFF00"
set object 55 rect from 0.032664, 40 to 34.079245, 50 fc rgb "#FF00FF"
set object 56 rect from 0.035063, 40 to 38.830018, 50 fc rgb "#808080"
set object 57 rect from 0.039934, 40 to 39.626534, 50 fc rgb "#800080"
set object 58 rect from 0.041146, 40 to 40.932586, 50 fc rgb "#008080"
set object 59 rect from 0.042077, 40 to 145.155726, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.157762, 50 to 163.735188, 60 fc rgb "#FF0000"
set object 61 rect from 0.130459, 50 to 128.767675, 60 fc rgb "#00FF00"
set object 62 rect from 0.132303, 50 to 129.419712, 60 fc rgb "#0000FF"
set object 63 rect from 0.132709, 50 to 132.581367, 60 fc rgb "#FFFF00"
set object 64 rect from 0.135968, 50 to 134.121686, 60 fc rgb "#FF00FF"
set object 65 rect from 0.137528, 50 to 138.645015, 60 fc rgb "#808080"
set object 66 rect from 0.142158, 50 to 139.248252, 60 fc rgb "#800080"
set object 67 rect from 0.143059, 50 to 139.958878, 60 fc rgb "#008080"
set object 68 rect from 0.143505, 50 to 153.498623, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
