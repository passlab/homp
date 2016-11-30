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

set object 15 rect from 1.228797, 0 to 161.106311, 10 fc rgb "#FF0000"
set object 16 rect from 0.319150, 0 to 37.778909, 10 fc rgb "#00FF00"
set object 17 rect from 0.321578, 0 to 37.858993, 10 fc rgb "#0000FF"
set object 18 rect from 0.322036, 0 to 37.955776, 10 fc rgb "#FFFF00"
set object 19 rect from 0.322898, 0 to 38.083252, 10 fc rgb "#FF00FF"
set object 20 rect from 0.323985, 0 to 54.390132, 10 fc rgb "#808080"
set object 21 rect from 0.462713, 0 to 54.473274, 10 fc rgb "#800080"
set object 22 rect from 0.463597, 0 to 54.539599, 10 fc rgb "#008080"
set object 23 rect from 0.463889, 0 to 144.431234, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.234524, 10 to 158.025475, 20 fc rgb "#FF0000"
set object 25 rect from 1.106138, 10 to 130.239477, 20 fc rgb "#00FF00"
set object 26 rect from 1.107788, 10 to 130.317092, 20 fc rgb "#0000FF"
set object 27 rect from 1.108255, 10 to 130.385181, 20 fc rgb "#FFFF00"
set object 28 rect from 1.108852, 10 to 130.488903, 20 fc rgb "#FF00FF"
set object 29 rect from 1.109741, 10 to 143.104961, 20 fc rgb "#808080"
set object 30 rect from 1.217031, 10 to 143.166230, 20 fc rgb "#800080"
set object 31 rect from 1.217732, 10 to 143.221970, 20 fc rgb "#008080"
set object 32 rect from 1.218000, 10 to 145.117302, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.231912, 20 to 167.310326, 30 fc rgb "#FF0000"
set object 34 rect from 0.688510, 20 to 81.122684, 30 fc rgb "#00FF00"
set object 35 rect from 0.690128, 20 to 81.192656, 30 fc rgb "#0000FF"
set object 36 rect from 0.690521, 20 to 81.511112, 30 fc rgb "#FFFF00"
set object 37 rect from 0.693268, 20 to 90.608866, 30 fc rgb "#FF00FF"
set object 38 rect from 0.770604, 20 to 103.096976, 30 fc rgb "#808080"
set object 39 rect from 0.876804, 20 to 103.639928, 30 fc rgb "#800080"
set object 40 rect from 0.881706, 20 to 104.586006, 30 fc rgb "#008080"
set object 41 rect from 0.889529, 20 to 144.818132, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.230467, 30 to 167.548463, 40 fc rgb "#FF0000"
set object 43 rect from 0.473809, 30 to 55.901976, 40 fc rgb "#00FF00"
set object 44 rect from 0.475674, 30 to 56.000640, 40 fc rgb "#0000FF"
set object 45 rect from 0.476300, 30 to 56.352612, 40 fc rgb "#FFFF00"
set object 46 rect from 0.479331, 30 to 65.805394, 40 fc rgb "#FF00FF"
set object 47 rect from 0.559696, 30 to 78.305735, 40 fc rgb "#808080"
set object 48 rect from 0.666023, 30 to 78.833165, 40 fc rgb "#800080"
set object 49 rect from 0.670727, 30 to 79.864971, 40 fc rgb "#008080"
set object 50 rect from 0.679271, 30 to 144.629504, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.233192, 40 to 167.393818, 50 fc rgb "#FF0000"
set object 52 rect from 0.897456, 40 to 105.686843, 50 fc rgb "#00FF00"
set object 53 rect from 0.899006, 40 to 105.770455, 50 fc rgb "#0000FF"
set object 54 rect from 0.899518, 40 to 106.087618, 50 fc rgb "#FFFF00"
set object 55 rect from 0.902219, 40 to 115.115048, 50 fc rgb "#FF00FF"
set object 56 rect from 0.978993, 40 to 127.616448, 50 fc rgb "#808080"
set object 57 rect from 1.085317, 40 to 128.133526, 50 fc rgb "#800080"
set object 58 rect from 1.089985, 40 to 129.060907, 50 fc rgb "#008080"
set object 59 rect from 1.097653, 40 to 144.974419, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.226903, 50 to 172.504641, 60 fc rgb "#FF0000"
set object 61 rect from 0.031205, 50 to 4.018450, 60 fc rgb "#00FF00"
set object 62 rect from 0.034546, 50 to 4.124993, 60 fc rgb "#0000FF"
set object 63 rect from 0.035188, 50 to 4.738620, 60 fc rgb "#FFFF00"
set object 64 rect from 0.040455, 50 to 18.660255, 60 fc rgb "#FF00FF"
set object 65 rect from 0.158791, 50 to 31.607234, 60 fc rgb "#808080"
set object 66 rect from 0.268908, 50 to 32.333874, 60 fc rgb "#800080"
set object 67 rect from 0.275545, 50 to 36.030458, 60 fc rgb "#008080"
set object 68 rect from 0.306539, 50 to 144.166638, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
