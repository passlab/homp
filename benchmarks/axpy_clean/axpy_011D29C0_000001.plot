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

set object 15 rect from 0.124003, 0 to 160.555719, 10 fc rgb "#FF0000"
set object 16 rect from 0.103889, 0 to 133.232161, 10 fc rgb "#00FF00"
set object 17 rect from 0.105911, 0 to 133.970631, 10 fc rgb "#0000FF"
set object 18 rect from 0.106247, 0 to 134.837870, 10 fc rgb "#FFFF00"
set object 19 rect from 0.106940, 0 to 136.164601, 10 fc rgb "#FF00FF"
set object 20 rect from 0.107988, 0 to 137.531727, 10 fc rgb "#808080"
set object 21 rect from 0.109073, 0 to 138.185617, 10 fc rgb "#800080"
set object 22 rect from 0.109872, 0 to 138.872340, 10 fc rgb "#008080"
set object 23 rect from 0.110133, 0 to 155.888812, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.122522, 10 to 159.114122, 20 fc rgb "#FF0000"
set object 25 rect from 0.089438, 10 to 115.010144, 20 fc rgb "#00FF00"
set object 26 rect from 0.091473, 10 to 115.792801, 20 fc rgb "#0000FF"
set object 27 rect from 0.091849, 10 to 116.723157, 20 fc rgb "#FFFF00"
set object 28 rect from 0.092629, 10 to 118.302353, 20 fc rgb "#FF00FF"
set object 29 rect from 0.093871, 10 to 119.714926, 20 fc rgb "#808080"
set object 30 rect from 0.094971, 10 to 120.497582, 20 fc rgb "#800080"
set object 31 rect from 0.095862, 10 to 121.306743, 20 fc rgb "#008080"
set object 32 rect from 0.096224, 10 to 153.938476, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.120906, 20 to 156.853253, 30 fc rgb "#FF0000"
set object 34 rect from 0.076206, 20 to 98.006294, 30 fc rgb "#00FF00"
set object 35 rect from 0.078011, 20 to 98.836927, 30 fc rgb "#0000FF"
set object 36 rect from 0.078418, 20 to 99.820291, 30 fc rgb "#FFFF00"
set object 37 rect from 0.079197, 20 to 101.394446, 30 fc rgb "#FF00FF"
set object 38 rect from 0.080442, 20 to 102.496476, 30 fc rgb "#808080"
set object 39 rect from 0.081319, 20 to 103.171829, 30 fc rgb "#800080"
set object 40 rect from 0.082130, 20 to 103.978469, 30 fc rgb "#008080"
set object 41 rect from 0.082491, 20 to 152.057579, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.119458, 30 to 155.118777, 40 fc rgb "#FF0000"
set object 43 rect from 0.061534, 30 to 79.655520, 40 fc rgb "#00FF00"
set object 44 rect from 0.063479, 30 to 80.498774, 40 fc rgb "#0000FF"
set object 45 rect from 0.063890, 30 to 81.615947, 40 fc rgb "#FFFF00"
set object 46 rect from 0.064806, 30 to 83.116882, 40 fc rgb "#FF00FF"
set object 47 rect from 0.065964, 30 to 84.212601, 40 fc rgb "#808080"
set object 48 rect from 0.066834, 30 to 84.957393, 40 fc rgb "#800080"
set object 49 rect from 0.067721, 30 to 85.796848, 40 fc rgb "#008080"
set object 50 rect from 0.068106, 30 to 150.118615, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.117901, 40 to 153.519370, 50 fc rgb "#FF0000"
set object 52 rect from 0.047071, 40 to 61.781911, 50 fc rgb "#00FF00"
set object 53 rect from 0.049325, 40 to 62.671872, 50 fc rgb "#0000FF"
set object 54 rect from 0.049766, 40 to 63.693111, 50 fc rgb "#FFFF00"
set object 55 rect from 0.050579, 40 to 65.484386, 50 fc rgb "#FF00FF"
set object 56 rect from 0.052007, 40 to 66.658366, 50 fc rgb "#808080"
set object 57 rect from 0.052941, 40 to 67.447343, 50 fc rgb "#800080"
set object 58 rect from 0.053823, 40 to 68.275437, 50 fc rgb "#008080"
set object 59 rect from 0.054225, 40 to 147.991554, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.116156, 50 to 154.155615, 60 fc rgb "#FF0000"
set object 61 rect from 0.026181, 50 to 36.394806, 60 fc rgb "#00FF00"
set object 62 rect from 0.029375, 50 to 38.175979, 60 fc rgb "#0000FF"
set object 63 rect from 0.030375, 50 to 40.600950, 60 fc rgb "#FFFF00"
set object 64 rect from 0.032336, 50 to 42.757044, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034029, 50 to 44.261769, 60 fc rgb "#808080"
set object 66 rect from 0.035223, 50 to 45.226201, 60 fc rgb "#800080"
set object 67 rect from 0.036361, 50 to 46.711984, 60 fc rgb "#008080"
set object 68 rect from 0.037145, 50 to 145.552692, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
