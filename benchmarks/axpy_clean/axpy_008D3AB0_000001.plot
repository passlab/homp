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

set object 15 rect from 1.047890, 0 to 156.950246, 10 fc rgb "#FF0000"
set object 16 rect from 0.547297, 0 to 76.052774, 10 fc rgb "#00FF00"
set object 17 rect from 0.549124, 0 to 76.144098, 10 fc rgb "#0000FF"
set object 18 rect from 0.549573, 0 to 76.230991, 10 fc rgb "#FFFF00"
set object 19 rect from 0.550215, 0 to 76.361534, 10 fc rgb "#FF00FF"
set object 20 rect from 0.551151, 0 to 87.811351, 10 fc rgb "#808080"
set object 21 rect from 0.633760, 0 to 87.886463, 10 fc rgb "#800080"
set object 22 rect from 0.634556, 0 to 87.960051, 10 fc rgb "#008080"
set object 23 rect from 0.634840, 0 to 145.128475, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.045788, 10 to 156.839246, 20 fc rgb "#FF0000"
set object 25 rect from 0.448477, 10 to 62.420285, 20 fc rgb "#00FF00"
set object 26 rect from 0.450749, 10 to 62.544315, 20 fc rgb "#0000FF"
set object 27 rect from 0.451422, 10 to 62.655598, 20 fc rgb "#FFFF00"
set object 28 rect from 0.452264, 10 to 62.821480, 20 fc rgb "#FF00FF"
set object 29 rect from 0.453460, 10 to 74.347378, 20 fc rgb "#808080"
set object 30 rect from 0.536708, 10 to 74.448544, 20 fc rgb "#800080"
set object 31 rect from 0.537600, 10 to 74.527396, 20 fc rgb "#008080"
set object 32 rect from 0.537929, 10 to 144.787009, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.041385, 20 to 167.294177, 30 fc rgb "#FF0000"
set object 34 rect from 0.045196, 20 to 6.759910, 30 fc rgb "#00FF00"
set object 35 rect from 0.049199, 20 to 6.956975, 30 fc rgb "#0000FF"
set object 36 rect from 0.050325, 20 to 7.768653, 30 fc rgb "#FFFF00"
set object 37 rect from 0.056227, 20 to 17.998251, 30 fc rgb "#FF00FF"
set object 38 rect from 0.129995, 20 to 29.321957, 30 fc rgb "#808080"
set object 39 rect from 0.211755, 20 to 29.857578, 30 fc rgb "#800080"
set object 40 rect from 0.215994, 20 to 34.799703, 30 fc rgb "#008080"
set object 41 rect from 0.251382, 20 to 144.194015, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.043342, 30 to 167.175828, 40 fc rgb "#FF0000"
set object 43 rect from 0.265728, 30 to 37.094351, 40 fc rgb "#00FF00"
set object 44 rect from 0.268032, 30 to 37.219491, 40 fc rgb "#0000FF"
set object 45 rect from 0.268688, 30 to 37.726286, 40 fc rgb "#FFFF00"
set object 46 rect from 0.272417, 30 to 47.979860, 40 fc rgb "#FF00FF"
set object 47 rect from 0.346361, 30 to 59.323107, 40 fc rgb "#808080"
set object 48 rect from 0.428204, 30 to 59.796781, 40 fc rgb "#800080"
set object 49 rect from 0.431875, 30 to 60.665830, 40 fc rgb "#008080"
set object 50 rect from 0.437991, 30 to 144.482820, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.049722, 40 to 167.757043, 50 fc rgb "#FF0000"
set object 52 rect from 0.643446, 40 to 89.374423, 50 fc rgb "#00FF00"
set object 53 rect from 0.645246, 40 to 89.462976, 50 fc rgb "#0000FF"
set object 54 rect from 0.645665, 40 to 89.893412, 50 fc rgb "#FFFF00"
set object 55 rect from 0.648796, 40 to 99.927472, 50 fc rgb "#FF00FF"
set object 56 rect from 0.721196, 40 to 111.243141, 50 fc rgb "#808080"
set object 57 rect from 0.802857, 40 to 111.758388, 50 fc rgb "#800080"
set object 58 rect from 0.806814, 40 to 112.604435, 50 fc rgb "#008080"
set object 59 rect from 0.812771, 40 to 145.394000, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.051142, 50 to 173.503639, 60 fc rgb "#FF0000"
set object 61 rect from 0.822815, 50 to 114.238599, 60 fc rgb "#00FF00"
set object 62 rect from 0.824671, 50 to 114.330340, 60 fc rgb "#0000FF"
set object 63 rect from 0.825109, 50 to 114.746780, 60 fc rgb "#FFFF00"
set object 64 rect from 0.828120, 50 to 130.279373, 60 fc rgb "#FF00FF"
set object 65 rect from 0.940199, 50 to 141.644654, 60 fc rgb "#808080"
set object 66 rect from 1.022429, 50 to 142.196905, 60 fc rgb "#800080"
set object 67 rect from 1.026520, 50 to 143.175296, 60 fc rgb "#008080"
set object 68 rect from 1.033364, 50 to 145.607695, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
