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

set object 15 rect from 0.247819, 0 to 156.408299, 10 fc rgb "#FF0000"
set object 16 rect from 0.078605, 0 to 47.633849, 10 fc rgb "#00FF00"
set object 17 rect from 0.080922, 0 to 48.132501, 10 fc rgb "#0000FF"
set object 18 rect from 0.081484, 0 to 48.628192, 10 fc rgb "#FFFF00"
set object 19 rect from 0.082360, 0 to 49.330918, 10 fc rgb "#FF00FF"
set object 20 rect from 0.083521, 0 to 57.570189, 10 fc rgb "#808080"
set object 21 rect from 0.097451, 0 to 57.923326, 10 fc rgb "#800080"
set object 22 rect from 0.098365, 0 to 58.305445, 10 fc rgb "#008080"
set object 23 rect from 0.098715, 0 to 146.156679, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.254280, 10 to 159.759257, 20 fc rgb "#FF0000"
set object 25 rect from 0.219605, 10 to 130.732820, 20 fc rgb "#00FF00"
set object 26 rect from 0.221355, 10 to 131.100153, 20 fc rgb "#0000FF"
set object 27 rect from 0.221743, 10 to 131.509485, 20 fc rgb "#FFFF00"
set object 28 rect from 0.222455, 10 to 132.110468, 20 fc rgb "#FF00FF"
set object 29 rect from 0.223463, 10 to 140.178789, 20 fc rgb "#808080"
set object 30 rect from 0.237096, 10 to 140.499392, 20 fc rgb "#800080"
set object 31 rect from 0.237928, 10 to 140.846020, 20 fc rgb "#008080"
set object 32 rect from 0.238261, 10 to 150.071944, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.251295, 20 to 164.147141, 30 fc rgb "#FF0000"
set object 34 rect from 0.145303, 20 to 86.756924, 30 fc rgb "#00FF00"
set object 35 rect from 0.147014, 20 to 87.130170, 30 fc rgb "#0000FF"
set object 36 rect from 0.147411, 20 to 88.808311, 30 fc rgb "#FFFF00"
set object 37 rect from 0.150273, 20 to 94.068099, 30 fc rgb "#FF00FF"
set object 38 rect from 0.159139, 20 to 102.137011, 30 fc rgb "#808080"
set object 39 rect from 0.172795, 20 to 102.685350, 30 fc rgb "#800080"
set object 40 rect from 0.173983, 20 to 103.042625, 30 fc rgb "#008080"
set object 41 rect from 0.174315, 20 to 148.279049, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.245850, 30 to 162.985405, 40 fc rgb "#FF0000"
set object 43 rect from 0.033024, 30 to 21.614699, 40 fc rgb "#00FF00"
set object 44 rect from 0.037041, 30 to 22.290216, 40 fc rgb "#0000FF"
set object 45 rect from 0.037804, 30 to 25.023625, 40 fc rgb "#FFFF00"
set object 46 rect from 0.042472, 30 to 30.911605, 40 fc rgb "#FF00FF"
set object 47 rect from 0.052383, 30 to 39.128399, 40 fc rgb "#808080"
set object 48 rect from 0.066289, 30 to 39.725239, 40 fc rgb "#800080"
set object 49 rect from 0.067712, 30 to 40.502494, 40 fc rgb "#008080"
set object 50 rect from 0.068610, 30 to 144.661316, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.249633, 40 to 163.530773, 50 fc rgb "#FF0000"
set object 52 rect from 0.107018, 40 to 64.223596, 50 fc rgb "#00FF00"
set object 53 rect from 0.108958, 40 to 64.628195, 50 fc rgb "#0000FF"
set object 54 rect from 0.109369, 40 to 66.534658, 50 fc rgb "#FFFF00"
set object 55 rect from 0.112606, 40 to 71.944104, 50 fc rgb "#FF00FF"
set object 56 rect from 0.121750, 40 to 80.018335, 50 fc rgb "#808080"
set object 57 rect from 0.135393, 40 to 80.535326, 50 fc rgb "#800080"
set object 58 rect from 0.136513, 40 to 80.945249, 50 fc rgb "#008080"
set object 59 rect from 0.136973, 40 to 147.239752, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.252752, 50 to 165.334911, 60 fc rgb "#FF0000"
set object 61 rect from 0.181770, 50 to 108.258641, 60 fc rgb "#00FF00"
set object 62 rect from 0.183367, 50 to 108.613554, 60 fc rgb "#0000FF"
set object 63 rect from 0.183730, 50 to 110.598693, 60 fc rgb "#FFFF00"
set object 64 rect from 0.187087, 50 to 115.957263, 60 fc rgb "#FF00FF"
set object 65 rect from 0.196155, 50 to 124.008427, 60 fc rgb "#808080"
set object 66 rect from 0.209776, 50 to 124.510037, 60 fc rgb "#800080"
set object 67 rect from 0.210871, 50 to 124.870864, 60 fc rgb "#008080"
set object 68 rect from 0.211217, 50 to 149.205368, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
