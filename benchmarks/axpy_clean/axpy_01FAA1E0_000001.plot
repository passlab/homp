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

set object 15 rect from 0.197601, 0 to 159.689791, 10 fc rgb "#FF0000"
set object 16 rect from 0.169056, 0 to 131.635525, 10 fc rgb "#00FF00"
set object 17 rect from 0.171376, 0 to 132.106611, 10 fc rgb "#0000FF"
set object 18 rect from 0.171734, 0 to 132.620034, 10 fc rgb "#FFFF00"
set object 19 rect from 0.172418, 0 to 133.430579, 10 fc rgb "#FF00FF"
set object 20 rect from 0.173458, 0 to 139.356105, 10 fc rgb "#808080"
set object 21 rect from 0.181172, 0 to 139.766381, 10 fc rgb "#800080"
set object 22 rect from 0.181952, 0 to 140.188205, 10 fc rgb "#008080"
set object 23 rect from 0.182262, 0 to 151.674396, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.196012, 10 to 159.087847, 20 fc rgb "#FF0000"
set object 25 rect from 0.145632, 10 to 113.576446, 20 fc rgb "#00FF00"
set object 26 rect from 0.147916, 10 to 114.155297, 20 fc rgb "#0000FF"
set object 27 rect from 0.148426, 10 to 114.813433, 20 fc rgb "#FFFF00"
set object 28 rect from 0.149313, 10 to 115.801020, 20 fc rgb "#FF00FF"
set object 29 rect from 0.150590, 10 to 121.937458, 20 fc rgb "#808080"
set object 30 rect from 0.158592, 10 to 122.443953, 20 fc rgb "#800080"
set object 31 rect from 0.159503, 10 to 122.941211, 20 fc rgb "#008080"
set object 32 rect from 0.159843, 10 to 150.402772, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.194359, 20 to 160.832095, 30 fc rgb "#FF0000"
set object 34 rect from 0.119528, 20 to 93.455981, 30 fc rgb "#00FF00"
set object 35 rect from 0.121762, 20 to 93.893197, 30 fc rgb "#0000FF"
set object 36 rect from 0.122094, 20 to 96.254787, 30 fc rgb "#FFFF00"
set object 37 rect from 0.125161, 20 to 98.754932, 30 fc rgb "#FF00FF"
set object 38 rect from 0.128415, 20 to 104.727413, 30 fc rgb "#808080"
set object 39 rect from 0.136176, 20 to 105.224671, 30 fc rgb "#800080"
set object 40 rect from 0.137083, 20 to 105.746561, 30 fc rgb "#008080"
set object 41 rect from 0.137497, 20 to 149.231214, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.192801, 30 to 159.873760, 40 fc rgb "#FF0000"
set object 43 rect from 0.092231, 30 to 72.364092, 40 fc rgb "#00FF00"
set object 44 rect from 0.094382, 30 to 72.870587, 40 fc rgb "#0000FF"
set object 45 rect from 0.094787, 30 to 75.313769, 40 fc rgb "#FFFF00"
set object 46 rect from 0.097965, 30 to 77.971712, 40 fc rgb "#FF00FF"
set object 47 rect from 0.101418, 30 to 83.881844, 40 fc rgb "#808080"
set object 48 rect from 0.109097, 30 to 84.403734, 40 fc rgb "#800080"
set object 49 rect from 0.110065, 30 to 84.924083, 40 fc rgb "#008080"
set object 50 rect from 0.110475, 30 to 147.995767, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.191161, 40 to 158.832294, 50 fc rgb "#FF0000"
set object 52 rect from 0.065146, 40 to 51.871068, 50 fc rgb "#00FF00"
set object 53 rect from 0.067778, 40 to 52.365247, 50 fc rgb "#0000FF"
set object 54 rect from 0.068149, 40 to 54.923892, 50 fc rgb "#FFFF00"
set object 55 rect from 0.071475, 40 to 57.572597, 50 fc rgb "#FF00FF"
set object 56 rect from 0.074933, 40 to 63.628982, 50 fc rgb "#808080"
set object 57 rect from 0.082820, 40 to 64.184741, 50 fc rgb "#800080"
set object 58 rect from 0.083804, 40 to 64.782835, 50 fc rgb "#008080"
set object 59 rect from 0.084303, 40 to 146.653324, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.189294, 50 to 159.403446, 60 fc rgb "#FF0000"
set object 61 rect from 0.030548, 50 to 26.186087, 60 fc rgb "#00FF00"
set object 62 rect from 0.034489, 50 to 27.165208, 60 fc rgb "#0000FF"
set object 63 rect from 0.035417, 50 to 30.676033, 60 fc rgb "#FFFF00"
set object 64 rect from 0.040029, 50 to 33.667276, 60 fc rgb "#FF00FF"
set object 65 rect from 0.043879, 50 to 39.933802, 60 fc rgb "#808080"
set object 66 rect from 0.052102, 50 to 40.654288, 60 fc rgb "#800080"
set object 67 rect from 0.053556, 50 to 41.776580, 60 fc rgb "#008080"
set object 68 rect from 0.054409, 50 to 144.945252, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
