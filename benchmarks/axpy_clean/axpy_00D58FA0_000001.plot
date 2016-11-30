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

set object 15 rect from 0.192094, 0 to 157.120439, 10 fc rgb "#FF0000"
set object 16 rect from 0.118693, 0 to 93.886907, 10 fc rgb "#00FF00"
set object 17 rect from 0.120613, 0 to 94.331137, 10 fc rgb "#0000FF"
set object 18 rect from 0.120940, 0 to 94.823762, 10 fc rgb "#FFFF00"
set object 19 rect from 0.121578, 0 to 95.560733, 10 fc rgb "#FF00FF"
set object 20 rect from 0.122516, 0 to 101.200501, 10 fc rgb "#808080"
set object 21 rect from 0.129744, 0 to 101.582260, 10 fc rgb "#800080"
set object 22 rect from 0.130482, 0 to 101.967928, 10 fc rgb "#008080"
set object 23 rect from 0.130741, 0 to 149.407124, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.190368, 10 to 156.238214, 20 fc rgb "#FF0000"
set object 25 rect from 0.097800, 10 to 77.695991, 20 fc rgb "#00FF00"
set object 26 rect from 0.099878, 10 to 78.242478, 20 fc rgb "#0000FF"
set object 27 rect from 0.100332, 10 to 78.810045, 20 fc rgb "#FFFF00"
set object 28 rect from 0.101103, 10 to 79.690665, 20 fc rgb "#FF00FF"
set object 29 rect from 0.102190, 10 to 85.459259, 20 fc rgb "#808080"
set object 30 rect from 0.109603, 10 to 85.908957, 20 fc rgb "#800080"
set object 31 rect from 0.110450, 10 to 86.396882, 20 fc rgb "#008080"
set object 32 rect from 0.110789, 10 to 147.987044, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.188585, 20 to 159.846616, 30 fc rgb "#FF0000"
set object 34 rect from 0.069996, 20 to 56.049533, 30 fc rgb "#00FF00"
set object 35 rect from 0.072165, 20 to 57.559400, 30 fc rgb "#0000FF"
set object 36 rect from 0.073842, 20 to 60.128690, 30 fc rgb "#FFFF00"
set object 37 rect from 0.077159, 20 to 63.294421, 30 fc rgb "#FF00FF"
set object 38 rect from 0.081202, 20 to 68.733559, 30 fc rgb "#808080"
set object 39 rect from 0.088166, 20 to 69.264435, 30 fc rgb "#800080"
set object 40 rect from 0.089163, 20 to 69.818740, 30 fc rgb "#008080"
set object 41 rect from 0.089555, 20 to 146.750421, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.194919, 30 to 162.789103, 40 fc rgb "#FF0000"
set object 43 rect from 0.163848, 30 to 128.927047, 40 fc rgb "#00FF00"
set object 44 rect from 0.165478, 30 to 129.355642, 40 fc rgb "#0000FF"
set object 45 rect from 0.165804, 30 to 131.682912, 40 fc rgb "#FFFF00"
set object 46 rect from 0.168826, 30 to 134.164766, 40 fc rgb "#FF00FF"
set object 47 rect from 0.171965, 30 to 139.600764, 40 fc rgb "#808080"
set object 48 rect from 0.178927, 30 to 140.100416, 40 fc rgb "#800080"
set object 49 rect from 0.179831, 30 to 140.631291, 40 fc rgb "#008080"
set object 50 rect from 0.180247, 30 to 151.817154, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.193548, 40 to 162.401876, 50 fc rgb "#FF0000"
set object 52 rect from 0.138510, 40 to 109.239362, 50 fc rgb "#00FF00"
set object 53 rect from 0.140280, 40 to 109.735105, 50 fc rgb "#0000FF"
set object 54 rect from 0.140673, 40 to 112.187294, 50 fc rgb "#FFFF00"
set object 55 rect from 0.143851, 40 to 115.171126, 50 fc rgb "#FF00FF"
set object 56 rect from 0.147648, 40 to 120.618059, 50 fc rgb "#808080"
set object 57 rect from 0.154623, 40 to 121.127855, 50 fc rgb "#800080"
set object 58 rect from 0.155549, 40 to 121.739931, 50 fc rgb "#008080"
set object 59 rect from 0.156053, 40 to 150.710894, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.186826, 50 to 161.649293, 60 fc rgb "#FF0000"
set object 61 rect from 0.034527, 50 to 29.559633, 60 fc rgb "#00FF00"
set object 62 rect from 0.038301, 50 to 32.712870, 60 fc rgb "#0000FF"
set object 63 rect from 0.042041, 50 to 36.195562, 60 fc rgb "#FFFF00"
set object 64 rect from 0.046521, 50 to 39.725905, 60 fc rgb "#FF00FF"
set object 65 rect from 0.051016, 50 to 45.398455, 60 fc rgb "#808080"
set object 66 rect from 0.058296, 50 to 46.044104, 60 fc rgb "#800080"
set object 67 rect from 0.059541, 50 to 46.977818, 60 fc rgb "#008080"
set object 68 rect from 0.060318, 50 to 144.890787, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
