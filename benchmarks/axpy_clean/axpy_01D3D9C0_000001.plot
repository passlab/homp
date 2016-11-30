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

set object 15 rect from 0.152825, 0 to 153.473076, 10 fc rgb "#FF0000"
set object 16 rect from 0.086772, 0 to 87.060390, 10 fc rgb "#00FF00"
set object 17 rect from 0.089066, 0 to 87.730572, 10 fc rgb "#0000FF"
set object 18 rect from 0.089536, 0 to 88.465557, 10 fc rgb "#FFFF00"
set object 19 rect from 0.090275, 0 to 89.437973, 10 fc rgb "#FF00FF"
set object 20 rect from 0.091260, 0 to 90.740114, 10 fc rgb "#808080"
set object 21 rect from 0.092608, 0 to 91.288641, 10 fc rgb "#800080"
set object 22 rect from 0.093398, 0 to 91.795934, 10 fc rgb "#008080"
set object 23 rect from 0.093678, 0 to 149.284070, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.151082, 10 to 152.386842, 20 fc rgb "#FF0000"
set object 25 rect from 0.069564, 10 to 70.478048, 20 fc rgb "#00FF00"
set object 26 rect from 0.072273, 10 to 71.407300, 20 fc rgb "#0000FF"
set object 27 rect from 0.072891, 10 to 72.235486, 20 fc rgb "#FFFF00"
set object 28 rect from 0.073788, 10 to 73.478759, 20 fc rgb "#FF00FF"
set object 29 rect from 0.075026, 10 to 74.888839, 20 fc rgb "#808080"
set object 30 rect from 0.076470, 10 to 75.519775, 20 fc rgb "#800080"
set object 31 rect from 0.077405, 10 to 76.162496, 20 fc rgb "#008080"
set object 32 rect from 0.077744, 10 to 147.420682, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.149032, 20 to 153.613417, 30 fc rgb "#FF0000"
set object 34 rect from 0.044325, 20 to 46.732542, 30 fc rgb "#00FF00"
set object 35 rect from 0.048120, 20 to 48.124930, 30 fc rgb "#0000FF"
set object 36 rect from 0.049172, 20 to 50.429930, 30 fc rgb "#FFFF00"
set object 37 rect from 0.051544, 20 to 52.766308, 30 fc rgb "#FF00FF"
set object 38 rect from 0.053897, 20 to 54.222505, 30 fc rgb "#808080"
set object 39 rect from 0.055401, 20 to 54.996707, 30 fc rgb "#800080"
set object 40 rect from 0.056665, 20 to 56.190909, 30 fc rgb "#008080"
set object 41 rect from 0.057387, 20 to 144.942997, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.157625, 30 to 158.408794, 40 fc rgb "#FF0000"
set object 43 rect from 0.135196, 30 to 134.207991, 40 fc rgb "#00FF00"
set object 44 rect from 0.137113, 30 to 134.823252, 40 fc rgb "#0000FF"
set object 45 rect from 0.137510, 30 to 135.771133, 40 fc rgb "#FFFF00"
set object 46 rect from 0.138479, 30 to 136.840726, 40 fc rgb "#FF00FF"
set object 47 rect from 0.139571, 30 to 138.013317, 40 fc rgb "#808080"
set object 48 rect from 0.140762, 30 to 138.603048, 40 fc rgb "#800080"
set object 49 rect from 0.141611, 30 to 139.181024, 40 fc rgb "#008080"
set object 50 rect from 0.141957, 30 to 154.216893, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.154580, 40 to 155.903737, 50 fc rgb "#FF0000"
set object 52 rect from 0.103102, 40 to 103.096136, 50 fc rgb "#00FF00"
set object 53 rect from 0.105431, 40 to 104.031296, 50 fc rgb "#0000FF"
set object 54 rect from 0.106134, 40 to 104.989002, 50 fc rgb "#FFFF00"
set object 55 rect from 0.107123, 40 to 106.255817, 50 fc rgb "#FF00FF"
set object 56 rect from 0.108397, 40 to 107.389162, 50 fc rgb "#808080"
set object 57 rect from 0.109578, 40 to 108.050541, 50 fc rgb "#800080"
set object 58 rect from 0.110483, 40 to 108.847318, 50 fc rgb "#008080"
set object 59 rect from 0.111058, 40 to 150.974803, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.156209, 50 to 156.896682, 60 fc rgb "#FF0000"
set object 61 rect from 0.119951, 50 to 119.393907, 60 fc rgb "#00FF00"
set object 62 rect from 0.122048, 50 to 119.982673, 60 fc rgb "#0000FF"
set object 63 rect from 0.122386, 50 to 120.905053, 60 fc rgb "#FFFF00"
set object 64 rect from 0.123331, 50 to 121.960873, 60 fc rgb "#FF00FF"
set object 65 rect from 0.124401, 50 to 123.079509, 60 fc rgb "#808080"
set object 66 rect from 0.125553, 50 to 123.720270, 60 fc rgb "#800080"
set object 67 rect from 0.126448, 50 to 124.410104, 60 fc rgb "#008080"
set object 68 rect from 0.126904, 50 to 152.678257, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
