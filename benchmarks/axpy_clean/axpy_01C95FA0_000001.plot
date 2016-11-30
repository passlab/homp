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

set object 15 rect from 0.190319, 0 to 155.869160, 10 fc rgb "#FF0000"
set object 16 rect from 0.095482, 0 to 75.625792, 10 fc rgb "#00FF00"
set object 17 rect from 0.097297, 0 to 76.149957, 10 fc rgb "#0000FF"
set object 18 rect from 0.097741, 0 to 76.665521, 10 fc rgb "#FFFF00"
set object 19 rect from 0.098400, 0 to 77.423677, 10 fc rgb "#FF00FF"
set object 20 rect from 0.099374, 0 to 83.182380, 10 fc rgb "#808080"
set object 21 rect from 0.106757, 0 to 83.573929, 10 fc rgb "#800080"
set object 22 rect from 0.107533, 0 to 83.975637, 10 fc rgb "#008080"
set object 23 rect from 0.107775, 0 to 148.007612, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.188853, 10 to 155.821576, 20 fc rgb "#FF0000"
set object 25 rect from 0.072548, 10 to 58.096233, 20 fc rgb "#00FF00"
set object 26 rect from 0.074820, 10 to 59.625794, 20 fc rgb "#0000FF"
set object 27 rect from 0.076572, 10 to 60.149959, 20 fc rgb "#FFFF00"
set object 28 rect from 0.077278, 10 to 61.121835, 20 fc rgb "#FF00FF"
set object 29 rect from 0.078500, 10 to 66.745575, 20 fc rgb "#808080"
set object 30 rect from 0.085715, 10 to 67.233849, 20 fc rgb "#800080"
set object 31 rect from 0.086575, 10 to 67.690927, 20 fc rgb "#008080"
set object 32 rect from 0.086910, 10 to 146.625452, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.186930, 20 to 161.244846, 30 fc rgb "#FF0000"
set object 34 rect from 0.036716, 20 to 31.592058, 30 fc rgb "#00FF00"
set object 35 rect from 0.040908, 20 to 34.499874, 30 fc rgb "#0000FF"
set object 36 rect from 0.044357, 20 to 37.874136, 30 fc rgb "#FFFF00"
set object 37 rect from 0.048735, 20 to 41.299074, 30 fc rgb "#FF00FF"
set object 38 rect from 0.053072, 20 to 47.001617, 30 fc rgb "#808080"
set object 39 rect from 0.060400, 20 to 47.696591, 30 fc rgb "#800080"
set object 40 rect from 0.061681, 20 to 48.606842, 30 fc rgb "#008080"
set object 41 rect from 0.062434, 20 to 144.946119, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.191881, 30 to 160.405587, 40 fc rgb "#FF0000"
set object 43 rect from 0.115938, 30 to 91.637505, 40 fc rgb "#00FF00"
set object 44 rect from 0.117820, 30 to 92.109391, 40 fc rgb "#0000FF"
set object 45 rect from 0.118202, 30 to 94.303520, 40 fc rgb "#FFFF00"
set object 46 rect from 0.121042, 30 to 96.928972, 40 fc rgb "#FF00FF"
set object 47 rect from 0.124423, 30 to 102.454453, 40 fc rgb "#808080"
set object 48 rect from 0.131478, 30 to 102.964554, 40 fc rgb "#800080"
set object 49 rect from 0.132384, 30 to 103.458291, 40 fc rgb "#008080"
set object 50 rect from 0.132755, 30 to 149.252461, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.193335, 40 to 161.534231, 50 fc rgb "#FF0000"
set object 52 rect from 0.139391, 40 to 109.750505, 50 fc rgb "#00FF00"
set object 53 rect from 0.141041, 40 to 110.208350, 50 fc rgb "#0000FF"
set object 54 rect from 0.141407, 40 to 112.342413, 50 fc rgb "#FFFF00"
set object 55 rect from 0.144142, 40 to 115.039624, 50 fc rgb "#FF00FF"
set object 56 rect from 0.147601, 40 to 120.497251, 50 fc rgb "#808080"
set object 57 rect from 0.154600, 40 to 121.010490, 50 fc rgb "#800080"
set object 58 rect from 0.155535, 40 to 121.492511, 50 fc rgb "#008080"
set object 59 rect from 0.155876, 40 to 150.407651, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.194728, 50 to 162.390645, 60 fc rgb "#FF0000"
set object 61 rect from 0.163660, 50 to 128.631005, 60 fc rgb "#00FF00"
set object 62 rect from 0.165249, 50 to 129.156727, 60 fc rgb "#0000FF"
set object 63 rect from 0.165700, 50 to 131.231514, 60 fc rgb "#FFFF00"
set object 64 rect from 0.168370, 50 to 133.741528, 60 fc rgb "#FF00FF"
set object 65 rect from 0.171578, 50 to 139.166378, 60 fc rgb "#808080"
set object 66 rect from 0.178532, 50 to 139.670250, 60 fc rgb "#800080"
set object 67 rect from 0.179439, 50 to 140.137464, 60 fc rgb "#008080"
set object 68 rect from 0.179776, 50 to 151.533948, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
