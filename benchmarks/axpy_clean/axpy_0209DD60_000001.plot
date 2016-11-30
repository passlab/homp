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

set object 15 rect from 0.190289, 0 to 154.504548, 10 fc rgb "#FF0000"
set object 16 rect from 0.045551, 0 to 37.154560, 10 fc rgb "#00FF00"
set object 17 rect from 0.049094, 0 to 38.184151, 10 fc rgb "#0000FF"
set object 18 rect from 0.050123, 0 to 39.428519, 10 fc rgb "#FFFF00"
set object 19 rect from 0.051716, 0 to 40.498612, 10 fc rgb "#FF00FF"
set object 20 rect from 0.053093, 0 to 46.503397, 10 fc rgb "#808080"
set object 21 rect from 0.060963, 0 to 46.904682, 10 fc rgb "#800080"
set object 22 rect from 0.061867, 0 to 47.434376, 10 fc rgb "#008080"
set object 23 rect from 0.062160, 0 to 144.759767, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.192011, 10 to 153.664506, 20 fc rgb "#FF0000"
set object 25 rect from 0.070985, 10 to 55.473103, 20 fc rgb "#00FF00"
set object 26 rect from 0.072877, 10 to 55.865983, 20 fc rgb "#0000FF"
set object 27 rect from 0.073186, 10 to 56.261926, 20 fc rgb "#FFFF00"
set object 28 rect from 0.073707, 10 to 56.982706, 20 fc rgb "#FF00FF"
set object 29 rect from 0.074656, 10 to 62.496774, 20 fc rgb "#808080"
set object 30 rect from 0.081892, 10 to 62.859071, 20 fc rgb "#800080"
set object 31 rect from 0.082606, 10 to 63.255014, 20 fc rgb "#008080"
set object 32 rect from 0.082866, 10 to 146.375613, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.194744, 20 to 160.586511, 30 fc rgb "#FF0000"
set object 34 rect from 0.117360, 20 to 90.609769, 30 fc rgb "#00FF00"
set object 35 rect from 0.118842, 20 to 91.049267, 30 fc rgb "#0000FF"
set object 36 rect from 0.119216, 20 to 93.114554, 30 fc rgb "#FFFF00"
set object 37 rect from 0.121923, 20 to 96.783465, 30 fc rgb "#FF00FF"
set object 38 rect from 0.126741, 20 to 102.366316, 30 fc rgb "#808080"
set object 39 rect from 0.134033, 20 to 102.821874, 30 fc rgb "#800080"
set object 40 rect from 0.134880, 20 to 103.239970, 30 fc rgb "#008080"
set object 41 rect from 0.135169, 20 to 148.501288, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.197332, 30 to 162.391917, 40 fc rgb "#FF0000"
set object 43 rect from 0.166852, 30 to 128.355937, 40 fc rgb "#00FF00"
set object 44 rect from 0.168220, 30 to 128.829069, 40 fc rgb "#0000FF"
set object 45 rect from 0.168643, 30 to 130.919584, 40 fc rgb "#FFFF00"
set object 46 rect from 0.171395, 30 to 134.435622, 40 fc rgb "#FF00FF"
set object 47 rect from 0.175979, 30 to 139.976433, 40 fc rgb "#808080"
set object 48 rect from 0.183233, 30 to 140.381534, 40 fc rgb "#800080"
set object 49 rect from 0.183986, 30 to 140.790473, 40 fc rgb "#008080"
set object 50 rect from 0.184296, 30 to 150.535242, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.193444, 40 to 160.466508, 50 fc rgb "#FF0000"
set object 52 rect from 0.090162, 40 to 69.949998, 50 fc rgb "#00FF00"
set object 53 rect from 0.091837, 40 to 70.571425, 50 fc rgb "#0000FF"
set object 54 rect from 0.092423, 40 to 72.681804, 50 fc rgb "#FFFF00"
set object 55 rect from 0.095209, 40 to 76.916332, 50 fc rgb "#FF00FF"
set object 56 rect from 0.100751, 40 to 82.553456, 50 fc rgb "#808080"
set object 57 rect from 0.108116, 40 to 83.070928, 50 fc rgb "#800080"
set object 58 rect from 0.109038, 40 to 83.761900, 50 fc rgb "#008080"
set object 59 rect from 0.109685, 40 to 147.390682, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.196141, 50 to 161.428068, 60 fc rgb "#FF0000"
set object 61 rect from 0.141647, 50 to 109.108693, 60 fc rgb "#00FF00"
set object 62 rect from 0.143044, 50 to 109.508452, 60 fc rgb "#0000FF"
set object 63 rect from 0.143367, 50 to 111.579855, 60 fc rgb "#FFFF00"
set object 64 rect from 0.146077, 50 to 115.178446, 60 fc rgb "#FF00FF"
set object 65 rect from 0.150788, 50 to 120.648937, 60 fc rgb "#808080"
set object 66 rect from 0.157948, 50 to 121.076213, 60 fc rgb "#800080"
set object 67 rect from 0.158739, 50 to 121.475972, 60 fc rgb "#008080"
set object 68 rect from 0.159025, 50 to 149.569855, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
