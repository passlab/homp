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

set object 15 rect from 0.195462, 0 to 154.974586, 10 fc rgb "#FF0000"
set object 16 rect from 0.073091, 0 to 56.376671, 10 fc rgb "#00FF00"
set object 17 rect from 0.075118, 0 to 57.238201, 10 fc rgb "#0000FF"
set object 18 rect from 0.076036, 0 to 57.703262, 10 fc rgb "#FFFF00"
set object 19 rect from 0.076683, 0 to 58.520326, 10 fc rgb "#FF00FF"
set object 20 rect from 0.077765, 0 to 64.148572, 10 fc rgb "#808080"
set object 21 rect from 0.085237, 0 to 64.556352, 10 fc rgb "#800080"
set object 22 rect from 0.086015, 0 to 64.983730, 10 fc rgb "#008080"
set object 23 rect from 0.086323, 0 to 146.754202, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.201390, 10 to 158.695852, 20 fc rgb "#FF0000"
set object 25 rect from 0.174638, 10 to 132.573168, 20 fc rgb "#00FF00"
set object 26 rect from 0.176174, 10 to 132.987731, 20 fc rgb "#0000FF"
set object 27 rect from 0.176824, 10 to 133.655548, 20 fc rgb "#FFFF00"
set object 28 rect from 0.177435, 10 to 134.340706, 20 fc rgb "#FF00FF"
set object 29 rect from 0.178347, 10 to 139.801622, 20 fc rgb "#808080"
set object 30 rect from 0.185582, 10 to 140.162666, 20 fc rgb "#800080"
set object 31 rect from 0.186287, 10 to 140.529741, 20 fc rgb "#008080"
set object 32 rect from 0.186539, 10 to 151.425189, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.198712, 20 to 162.071887, 30 fc rgb "#FF0000"
set object 34 rect from 0.122846, 20 to 93.480725, 30 fc rgb "#00FF00"
set object 35 rect from 0.124315, 20 to 93.899804, 30 fc rgb "#0000FF"
set object 36 rect from 0.124676, 20 to 96.064575, 30 fc rgb "#FFFF00"
set object 37 rect from 0.127547, 20 to 100.378281, 30 fc rgb "#FF00FF"
set object 38 rect from 0.133269, 20 to 105.806782, 30 fc rgb "#808080"
set object 39 rect from 0.140474, 20 to 106.218324, 30 fc rgb "#800080"
set object 40 rect from 0.141250, 20 to 106.648724, 30 fc rgb "#008080"
set object 41 rect from 0.141590, 20 to 149.460163, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.193519, 30 to 160.431728, 40 fc rgb "#FF0000"
set object 43 rect from 0.038002, 30 to 31.194667, 40 fc rgb "#00FF00"
set object 44 rect from 0.041864, 30 to 32.300409, 40 fc rgb "#0000FF"
set object 45 rect from 0.042973, 30 to 35.681735, 40 fc rgb "#FFFF00"
set object 46 rect from 0.047493, 30 to 40.080611, 40 fc rgb "#FF00FF"
set object 47 rect from 0.053291, 30 to 45.920665, 40 fc rgb "#808080"
set object 48 rect from 0.061038, 30 to 46.427182, 40 fc rgb "#800080"
set object 49 rect from 0.062091, 30 to 47.486200, 40 fc rgb "#008080"
set object 50 rect from 0.063112, 30 to 144.906761, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.197306, 40 to 160.678961, 50 fc rgb "#FF0000"
set object 52 rect from 0.094287, 40 to 72.255898, 50 fc rgb "#00FF00"
set object 53 rect from 0.096171, 40 to 72.711166, 50 fc rgb "#0000FF"
set object 54 rect from 0.096566, 40 to 74.854832, 50 fc rgb "#FFFF00"
set object 55 rect from 0.099432, 40 to 78.808247, 50 fc rgb "#FF00FF"
set object 56 rect from 0.104670, 40 to 84.257852, 50 fc rgb "#808080"
set object 57 rect from 0.111904, 40 to 84.719139, 50 fc rgb "#800080"
set object 58 rect from 0.112725, 40 to 85.219637, 50 fc rgb "#008080"
set object 59 rect from 0.113177, 40 to 148.147880, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.200018, 50 to 162.586716, 60 fc rgb "#FF0000"
set object 61 rect from 0.149191, 50 to 113.286231, 60 fc rgb "#00FF00"
set object 62 rect from 0.150594, 50 to 113.673659, 60 fc rgb "#0000FF"
set object 63 rect from 0.150909, 50 to 115.941694, 60 fc rgb "#FFFF00"
set object 64 rect from 0.153921, 50 to 119.735315, 60 fc rgb "#FF00FF"
set object 65 rect from 0.158950, 50 to 125.163063, 60 fc rgb "#808080"
set object 66 rect from 0.166154, 50 to 125.558779, 60 fc rgb "#800080"
set object 67 rect from 0.166888, 50 to 126.000468, 60 fc rgb "#008080"
set object 68 rect from 0.167263, 50 to 150.460383, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
