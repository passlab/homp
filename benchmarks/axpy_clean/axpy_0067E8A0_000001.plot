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

set object 15 rect from 0.141290, 0 to 159.716291, 10 fc rgb "#FF0000"
set object 16 rect from 0.120136, 0 to 133.751204, 10 fc rgb "#00FF00"
set object 17 rect from 0.122003, 0 to 134.437391, 10 fc rgb "#0000FF"
set object 18 rect from 0.122373, 0 to 135.253332, 10 fc rgb "#FFFF00"
set object 19 rect from 0.123122, 0 to 136.376086, 10 fc rgb "#FF00FF"
set object 20 rect from 0.124138, 0 to 138.334568, 10 fc rgb "#808080"
set object 21 rect from 0.125933, 0 to 138.926190, 10 fc rgb "#800080"
set object 22 rect from 0.126738, 0 to 139.551888, 10 fc rgb "#008080"
set object 23 rect from 0.127052, 0 to 154.699661, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.139725, 10 to 158.465978, 20 fc rgb "#FF0000"
set object 25 rect from 0.105036, 10 to 117.266255, 20 fc rgb "#00FF00"
set object 26 rect from 0.107020, 10 to 118.044811, 20 fc rgb "#0000FF"
set object 27 rect from 0.107468, 10 to 118.865159, 20 fc rgb "#FFFF00"
set object 28 rect from 0.108242, 10 to 120.163852, 20 fc rgb "#FF00FF"
set object 29 rect from 0.109422, 10 to 122.319172, 20 fc rgb "#808080"
set object 30 rect from 0.111397, 10 to 123.010857, 20 fc rgb "#800080"
set object 31 rect from 0.112289, 10 to 123.712438, 20 fc rgb "#008080"
set object 32 rect from 0.112628, 10 to 152.897323, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137937, 20 to 157.361936, 30 fc rgb "#FF0000"
set object 34 rect from 0.089794, 20 to 100.293056, 30 fc rgb "#00FF00"
set object 35 rect from 0.091583, 20 to 100.948453, 30 fc rgb "#0000FF"
set object 36 rect from 0.091920, 20 to 103.103781, 30 fc rgb "#FFFF00"
set object 37 rect from 0.093890, 20 to 104.447561, 30 fc rgb "#FF00FF"
set object 38 rect from 0.095099, 20 to 106.137736, 30 fc rgb "#808080"
set object 39 rect from 0.096662, 20 to 106.821719, 30 fc rgb "#800080"
set object 40 rect from 0.097566, 20 to 107.557391, 30 fc rgb "#008080"
set object 41 rect from 0.097939, 20 to 151.199455, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.136482, 30 to 155.882872, 40 fc rgb "#FF0000"
set object 43 rect from 0.073247, 30 to 82.189416, 40 fc rgb "#00FF00"
set object 44 rect from 0.075122, 30 to 82.960270, 40 fc rgb "#0000FF"
set object 45 rect from 0.075562, 30 to 85.073814, 40 fc rgb "#FFFF00"
set object 46 rect from 0.077487, 30 to 86.367010, 40 fc rgb "#FF00FF"
set object 47 rect from 0.078659, 30 to 88.054981, 40 fc rgb "#808080"
set object 48 rect from 0.080220, 30 to 88.806048, 40 fc rgb "#800080"
set object 49 rect from 0.081177, 30 to 89.526326, 40 fc rgb "#008080"
set object 50 rect from 0.081547, 30 to 149.435600, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.134929, 40 to 154.818411, 50 fc rgb "#FF0000"
set object 52 rect from 0.054041, 40 to 61.756698, 50 fc rgb "#00FF00"
set object 53 rect from 0.056619, 40 to 62.685908, 50 fc rgb "#0000FF"
set object 54 rect from 0.057126, 40 to 65.003983, 50 fc rgb "#FFFF00"
set object 55 rect from 0.059277, 40 to 66.638077, 50 fc rgb "#FF00FF"
set object 56 rect from 0.060746, 40 to 68.443708, 50 fc rgb "#808080"
set object 57 rect from 0.062369, 40 to 69.184879, 50 fc rgb "#800080"
set object 58 rect from 0.063364, 40 to 70.176766, 50 fc rgb "#008080"
set object 59 rect from 0.063959, 40 to 147.655260, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.133182, 50 to 155.628879, 60 fc rgb "#FF0000"
set object 61 rect from 0.014785, 50 to 19.131822, 60 fc rgb "#00FF00"
set object 62 rect from 0.018040, 50 to 20.842880, 60 fc rgb "#0000FF"
set object 63 rect from 0.019101, 50 to 24.850033, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022768, 50 to 26.709551, 60 fc rgb "#FF00FF"
set object 65 rect from 0.024424, 50 to 28.652646, 60 fc rgb "#808080"
set object 66 rect from 0.026219, 50 to 29.548862, 60 fc rgb "#800080"
set object 67 rect from 0.027444, 50 to 30.845359, 60 fc rgb "#008080"
set object 68 rect from 0.028196, 50 to 145.297595, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
