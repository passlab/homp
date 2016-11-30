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

set object 15 rect from 0.129112, 0 to 160.041769, 10 fc rgb "#FF0000"
set object 16 rect from 0.108175, 0 to 133.753064, 10 fc rgb "#00FF00"
set object 17 rect from 0.110422, 0 to 134.485793, 10 fc rgb "#0000FF"
set object 18 rect from 0.110789, 0 to 135.367935, 10 fc rgb "#FFFF00"
set object 19 rect from 0.111520, 0 to 136.607352, 10 fc rgb "#FF00FF"
set object 20 rect from 0.112540, 0 to 137.121319, 10 fc rgb "#808080"
set object 21 rect from 0.112964, 0 to 137.749538, 10 fc rgb "#800080"
set object 22 rect from 0.113735, 0 to 138.370441, 10 fc rgb "#008080"
set object 23 rect from 0.114014, 0 to 156.283395, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.127646, 10 to 158.589639, 20 fc rgb "#FF0000"
set object 25 rect from 0.093483, 10 to 115.840111, 20 fc rgb "#00FF00"
set object 26 rect from 0.095774, 10 to 116.716169, 20 fc rgb "#0000FF"
set object 27 rect from 0.096168, 10 to 117.657881, 20 fc rgb "#FFFF00"
set object 28 rect from 0.096978, 10 to 119.095346, 20 fc rgb "#FF00FF"
set object 29 rect from 0.098153, 10 to 119.665225, 20 fc rgb "#808080"
set object 30 rect from 0.098615, 10 to 120.372424, 20 fc rgb "#800080"
set object 31 rect from 0.099431, 10 to 121.067455, 20 fc rgb "#008080"
set object 32 rect from 0.099772, 10 to 154.239621, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.125793, 20 to 156.102331, 30 fc rgb "#FF0000"
set object 34 rect from 0.079254, 20 to 98.465423, 30 fc rgb "#00FF00"
set object 35 rect from 0.081395, 20 to 99.173817, 30 fc rgb "#0000FF"
set object 36 rect from 0.081751, 20 to 100.187230, 30 fc rgb "#FFFF00"
set object 37 rect from 0.082564, 20 to 101.472782, 30 fc rgb "#FF00FF"
set object 38 rect from 0.083620, 20 to 101.950319, 30 fc rgb "#808080"
set object 39 rect from 0.084017, 20 to 102.585816, 30 fc rgb "#800080"
set object 40 rect from 0.084805, 20 to 103.280847, 30 fc rgb "#008080"
set object 41 rect from 0.085117, 20 to 152.250492, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.124266, 30 to 154.458346, 40 fc rgb "#FF0000"
set object 43 rect from 0.064244, 30 to 80.213446, 40 fc rgb "#00FF00"
set object 44 rect from 0.066379, 30 to 80.947369, 40 fc rgb "#0000FF"
set object 45 rect from 0.066749, 30 to 82.081045, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067665, 30 to 83.389701, 40 fc rgb "#FF00FF"
set object 47 rect from 0.068754, 30 to 83.928003, 40 fc rgb "#808080"
set object 48 rect from 0.069197, 30 to 84.586568, 40 fc rgb "#800080"
set object 49 rect from 0.070016, 30 to 85.359384, 40 fc rgb "#008080"
set object 50 rect from 0.070382, 30 to 150.295402, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.122652, 40 to 152.663679, 50 fc rgb "#FF0000"
set object 52 rect from 0.049484, 40 to 62.740332, 50 fc rgb "#00FF00"
set object 53 rect from 0.052023, 40 to 63.572682, 50 fc rgb "#0000FF"
set object 54 rect from 0.052431, 40 to 64.641971, 50 fc rgb "#FFFF00"
set object 55 rect from 0.053340, 40 to 66.101308, 50 fc rgb "#FF00FF"
set object 56 rect from 0.054522, 40 to 66.651742, 50 fc rgb "#808080"
set object 57 rect from 0.054969, 40 to 67.300601, 50 fc rgb "#800080"
set object 58 rect from 0.055803, 40 to 68.094059, 50 fc rgb "#008080"
set object 59 rect from 0.056187, 40 to 148.193289, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.120766, 50 to 153.340386, 60 fc rgb "#FF0000"
set object 61 rect from 0.028115, 50 to 37.762781, 60 fc rgb "#00FF00"
set object 62 rect from 0.031637, 50 to 39.332676, 60 fc rgb "#0000FF"
set object 63 rect from 0.032499, 50 to 41.902621, 60 fc rgb "#FFFF00"
set object 64 rect from 0.034635, 50 to 43.789666, 60 fc rgb "#FF00FF"
set object 65 rect from 0.036165, 50 to 44.624442, 60 fc rgb "#808080"
set object 66 rect from 0.036865, 50 to 45.526031, 60 fc rgb "#800080"
set object 67 rect from 0.038033, 50 to 46.851706, 60 fc rgb "#008080"
set object 68 rect from 0.038695, 50 to 145.448401, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
