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

set object 15 rect from 0.137911, 0 to 159.824221, 10 fc rgb "#FF0000"
set object 16 rect from 0.116125, 0 to 133.432823, 10 fc rgb "#00FF00"
set object 17 rect from 0.118331, 0 to 134.112563, 10 fc rgb "#0000FF"
set object 18 rect from 0.118691, 0 to 134.955176, 10 fc rgb "#FFFF00"
set object 19 rect from 0.119437, 0 to 136.102030, 10 fc rgb "#FF00FF"
set object 20 rect from 0.120447, 0 to 137.494340, 10 fc rgb "#808080"
set object 21 rect from 0.121691, 0 to 138.077947, 10 fc rgb "#800080"
set object 22 rect from 0.122455, 0 to 138.673991, 10 fc rgb "#008080"
set object 23 rect from 0.122734, 0 to 155.346454, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.136311, 10 to 158.479438, 20 fc rgb "#FF0000"
set object 25 rect from 0.100508, 10 to 115.845345, 20 fc rgb "#00FF00"
set object 26 rect from 0.102863, 10 to 116.767136, 20 fc rgb "#0000FF"
set object 27 rect from 0.103373, 10 to 117.645950, 20 fc rgb "#FFFF00"
set object 28 rect from 0.104165, 10 to 118.967003, 20 fc rgb "#FF00FF"
set object 29 rect from 0.105320, 10 to 120.455413, 20 fc rgb "#808080"
set object 30 rect from 0.106642, 10 to 121.129523, 20 fc rgb "#800080"
set object 31 rect from 0.107473, 10 to 121.801341, 20 fc rgb "#008080"
set object 32 rect from 0.107840, 10 to 153.505130, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.134635, 20 to 156.737618, 30 fc rgb "#FF0000"
set object 34 rect from 0.084145, 20 to 97.264852, 30 fc rgb "#00FF00"
set object 35 rect from 0.086396, 20 to 98.072409, 30 fc rgb "#0000FF"
set object 36 rect from 0.086822, 20 to 99.082412, 30 fc rgb "#FFFF00"
set object 37 rect from 0.087720, 20 to 100.489418, 30 fc rgb "#FF00FF"
set object 38 rect from 0.088964, 20 to 101.890762, 30 fc rgb "#808080"
set object 39 rect from 0.090213, 20 to 102.581793, 30 fc rgb "#800080"
set object 40 rect from 0.091113, 20 to 103.457237, 30 fc rgb "#008080"
set object 41 rect from 0.091601, 20 to 151.748647, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133105, 30 to 154.571693, 40 fc rgb "#FF0000"
set object 43 rect from 0.050491, 30 to 59.047482, 40 fc rgb "#00FF00"
set object 44 rect from 0.052589, 30 to 59.739660, 40 fc rgb "#0000FF"
set object 45 rect from 0.052929, 30 to 60.684068, 40 fc rgb "#FFFF00"
set object 46 rect from 0.053786, 30 to 61.939527, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054874, 30 to 63.248108, 40 fc rgb "#808080"
set object 48 rect from 0.056035, 30 to 63.872433, 40 fc rgb "#800080"
set object 49 rect from 0.056842, 30 to 64.554431, 40 fc rgb "#008080"
set object 50 rect from 0.057293, 30 to 149.927716, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.131423, 40 to 152.871771, 50 fc rgb "#FF0000"
set object 52 rect from 0.036206, 40 to 43.329608, 50 fc rgb "#00FF00"
set object 53 rect from 0.038664, 40 to 44.074941, 50 fc rgb "#0000FF"
set object 54 rect from 0.039101, 40 to 45.058956, 50 fc rgb "#FFFF00"
set object 55 rect from 0.039958, 40 to 46.427502, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041181, 40 to 47.772285, 50 fc rgb "#808080"
set object 57 rect from 0.042366, 40 to 48.409048, 50 fc rgb "#800080"
set object 58 rect from 0.043176, 40 to 49.103484, 50 fc rgb "#008080"
set object 59 rect from 0.043574, 40 to 147.907677, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.129478, 50 to 154.498245, 60 fc rgb "#FF0000"
set object 61 rect from 0.015482, 50 to 20.670610, 60 fc rgb "#00FF00"
set object 62 rect from 0.018734, 50 to 22.027864, 60 fc rgb "#0000FF"
set object 63 rect from 0.019617, 50 to 25.259206, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022499, 50 to 27.140102, 60 fc rgb "#FF00FF"
set object 65 rect from 0.024141, 50 to 28.838912, 60 fc rgb "#808080"
set object 66 rect from 0.025642, 50 to 29.753894, 60 fc rgb "#800080"
set object 67 rect from 0.026844, 50 to 31.124699, 60 fc rgb "#008080"
set object 68 rect from 0.027656, 50 to 145.326682, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
