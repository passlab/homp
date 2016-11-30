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

set object 15 rect from 0.222664, 0 to 157.666200, 10 fc rgb "#FF0000"
set object 16 rect from 0.135517, 0 to 92.419187, 10 fc rgb "#00FF00"
set object 17 rect from 0.137695, 0 to 92.889089, 10 fc rgb "#0000FF"
set object 18 rect from 0.138109, 0 to 93.355620, 10 fc rgb "#FFFF00"
set object 19 rect from 0.138805, 0 to 94.083365, 10 fc rgb "#FF00FF"
set object 20 rect from 0.139882, 0 to 100.329424, 10 fc rgb "#808080"
set object 21 rect from 0.149178, 0 to 100.747492, 10 fc rgb "#800080"
set object 22 rect from 0.150163, 0 to 101.209981, 10 fc rgb "#008080"
set object 23 rect from 0.150482, 0 to 149.387711, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.218552, 10 to 155.161852, 20 fc rgb "#FF0000"
set object 25 rect from 0.079859, 10 to 55.154223, 20 fc rgb "#00FF00"
set object 26 rect from 0.082334, 10 to 55.655764, 20 fc rgb "#0000FF"
set object 27 rect from 0.082791, 10 to 56.176828, 20 fc rgb "#FFFF00"
set object 28 rect from 0.083844, 10 to 57.233099, 20 fc rgb "#FF00FF"
set object 29 rect from 0.085160, 10 to 63.483872, 20 fc rgb "#808080"
set object 30 rect from 0.094483, 10 to 63.946371, 20 fc rgb "#800080"
set object 31 rect from 0.095472, 10 to 64.441853, 20 fc rgb "#008080"
set object 32 rect from 0.095861, 10 to 146.446456, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.216008, 20 to 159.322995, 30 fc rgb "#FF0000"
set object 34 rect from 0.042884, 20 to 31.133347, 30 fc rgb "#00FF00"
set object 35 rect from 0.047125, 20 to 32.298682, 30 fc rgb "#0000FF"
set object 36 rect from 0.048120, 20 to 35.151750, 30 fc rgb "#FFFF00"
set object 37 rect from 0.052382, 20 to 39.149287, 30 fc rgb "#FF00FF"
set object 38 rect from 0.058287, 20 to 45.379185, 30 fc rgb "#808080"
set object 39 rect from 0.067648, 20 to 46.002581, 30 fc rgb "#800080"
set object 40 rect from 0.068925, 20 to 46.866315, 30 fc rgb "#008080"
set object 41 rect from 0.069753, 20 to 144.850944, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.226174, 30 to 164.119602, 40 fc rgb "#FF0000"
set object 43 rect from 0.188526, 30 to 127.907548, 40 fc rgb "#00FF00"
set object 44 rect from 0.190356, 30 to 128.407064, 40 fc rgb "#0000FF"
set object 45 rect from 0.190858, 30 to 130.290042, 40 fc rgb "#FFFF00"
set object 46 rect from 0.193655, 30 to 133.748329, 40 fc rgb "#FF00FF"
set object 47 rect from 0.198795, 30 to 139.797135, 40 fc rgb "#808080"
set object 48 rect from 0.207778, 30 to 140.252230, 40 fc rgb "#800080"
set object 49 rect from 0.208752, 30 to 140.728192, 40 fc rgb "#008080"
set object 50 rect from 0.209164, 30 to 151.867823, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.224512, 40 to 163.345421, 50 fc rgb "#FF0000"
set object 52 rect from 0.158292, 40 to 107.553657, 50 fc rgb "#00FF00"
set object 53 rect from 0.160146, 40 to 108.065301, 50 fc rgb "#0000FF"
set object 54 rect from 0.160642, 40 to 110.154279, 50 fc rgb "#FFFF00"
set object 55 rect from 0.163757, 40 to 113.749227, 50 fc rgb "#FF00FF"
set object 56 rect from 0.169088, 40 to 119.776495, 50 fc rgb "#808080"
set object 57 rect from 0.178071, 40 to 120.288129, 50 fc rgb "#800080"
set object 58 rect from 0.179115, 40 to 120.838143, 50 fc rgb "#008080"
set object 59 rect from 0.179627, 40 to 150.779240, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.220815, 50 to 161.105637, 60 fc rgb "#FF0000"
set object 61 rect from 0.104446, 50 to 71.494416, 60 fc rgb "#00FF00"
set object 62 rect from 0.106596, 50 to 72.082129, 60 fc rgb "#0000FF"
set object 63 rect from 0.107202, 50 to 74.243144, 60 fc rgb "#FFFF00"
set object 64 rect from 0.110439, 50 to 77.993602, 60 fc rgb "#FF00FF"
set object 65 rect from 0.115996, 50 to 84.018172, 60 fc rgb "#808080"
set object 66 rect from 0.124965, 50 to 84.530488, 60 fc rgb "#800080"
set object 67 rect from 0.125992, 50 to 85.095319, 60 fc rgb "#008080"
set object 68 rect from 0.126529, 50 to 148.130828, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
