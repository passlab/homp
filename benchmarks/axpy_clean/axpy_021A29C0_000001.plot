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

set object 15 rect from 0.227788, 0 to 155.529718, 10 fc rgb "#FF0000"
set object 16 rect from 0.201076, 0 to 136.254299, 10 fc rgb "#00FF00"
set object 17 rect from 0.203398, 0 to 136.776355, 10 fc rgb "#0000FF"
set object 18 rect from 0.203941, 0 to 137.358145, 10 fc rgb "#FFFF00"
set object 19 rect from 0.204814, 0 to 138.121780, 10 fc rgb "#FF00FF"
set object 20 rect from 0.205961, 0 to 139.018264, 10 fc rgb "#808080"
set object 21 rect from 0.207318, 0 to 139.425589, 10 fc rgb "#800080"
set object 22 rect from 0.208150, 0 to 139.781899, 10 fc rgb "#008080"
set object 23 rect from 0.208459, 0 to 152.412724, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.224115, 10 to 153.044810, 20 fc rgb "#FF0000"
set object 25 rect from 0.157629, 10 to 107.140238, 20 fc rgb "#00FF00"
set object 26 rect from 0.160008, 10 to 107.658274, 20 fc rgb "#0000FF"
set object 27 rect from 0.160557, 10 to 108.182349, 20 fc rgb "#FFFF00"
set object 28 rect from 0.161367, 10 to 109.013759, 20 fc rgb "#FF00FF"
set object 29 rect from 0.162605, 10 to 109.909602, 20 fc rgb "#808080"
set object 30 rect from 0.163935, 10 to 110.326967, 20 fc rgb "#800080"
set object 31 rect from 0.164783, 10 to 110.711474, 20 fc rgb "#008080"
set object 32 rect from 0.165116, 10 to 149.919176, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.221900, 20 to 151.537038, 30 fc rgb "#FF0000"
set object 34 rect from 0.140394, 20 to 95.532746, 30 fc rgb "#00FF00"
set object 35 rect from 0.142734, 20 to 95.988368, 30 fc rgb "#0000FF"
set object 36 rect from 0.143171, 20 to 96.580897, 30 fc rgb "#FFFF00"
set object 37 rect from 0.144041, 20 to 97.457943, 30 fc rgb "#FF00FF"
set object 38 rect from 0.145350, 20 to 98.242376, 30 fc rgb "#808080"
set object 39 rect from 0.146535, 20 to 98.655061, 30 fc rgb "#800080"
set object 40 rect from 0.147398, 20 to 99.074445, 30 fc rgb "#008080"
set object 41 rect from 0.147773, 20 to 148.504636, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.219639, 30 to 150.241929, 40 fc rgb "#FF0000"
set object 43 rect from 0.122817, 30 to 83.838682, 40 fc rgb "#00FF00"
set object 44 rect from 0.125313, 30 to 84.414413, 40 fc rgb "#0000FF"
set object 45 rect from 0.125911, 30 to 85.046539, 40 fc rgb "#FFFF00"
set object 46 rect from 0.126871, 30 to 86.003418, 40 fc rgb "#FF00FF"
set object 47 rect from 0.128291, 30 to 86.801950, 40 fc rgb "#808080"
set object 48 rect from 0.129472, 30 to 87.197196, 40 fc rgb "#800080"
set object 49 rect from 0.130339, 30 to 87.670936, 40 fc rgb "#008080"
set object 50 rect from 0.130787, 30 to 146.897513, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.217356, 40 to 150.544563, 50 fc rgb "#FF0000"
set object 52 rect from 0.098481, 40 to 68.125800, 50 fc rgb "#00FF00"
set object 53 rect from 0.101990, 40 to 68.994126, 50 fc rgb "#0000FF"
set object 54 rect from 0.102952, 40 to 70.412685, 50 fc rgb "#FFFF00"
set object 55 rect from 0.105130, 40 to 71.882900, 50 fc rgb "#FF00FF"
set object 56 rect from 0.107254, 40 to 72.892134, 50 fc rgb "#808080"
set object 57 rect from 0.108772, 40 to 73.443727, 50 fc rgb "#800080"
set object 58 rect from 0.110072, 40 to 74.367068, 50 fc rgb "#008080"
set object 59 rect from 0.110960, 40 to 145.023993, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.226032, 50 to 154.229169, 60 fc rgb "#FF0000"
set object 61 rect from 0.172487, 50 to 116.920526, 60 fc rgb "#00FF00"
set object 62 rect from 0.174601, 50 to 117.329851, 60 fc rgb "#0000FF"
set object 63 rect from 0.174975, 50 to 117.928420, 60 fc rgb "#FFFF00"
set object 64 rect from 0.175854, 50 to 118.710833, 60 fc rgb "#FF00FF"
set object 65 rect from 0.177040, 50 to 119.551642, 60 fc rgb "#808080"
set object 66 rect from 0.178296, 50 to 119.980425, 60 fc rgb "#800080"
set object 67 rect from 0.179183, 50 to 120.397809, 60 fc rgb "#008080"
set object 68 rect from 0.179555, 50 to 151.202867, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
