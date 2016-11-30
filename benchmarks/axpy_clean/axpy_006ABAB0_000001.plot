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

set object 15 rect from 0.137986, 0 to 158.596806, 10 fc rgb "#FF0000"
set object 16 rect from 0.117215, 0 to 134.139957, 10 fc rgb "#00FF00"
set object 17 rect from 0.119370, 0 to 134.856769, 10 fc rgb "#0000FF"
set object 18 rect from 0.119774, 0 to 135.730163, 10 fc rgb "#FFFF00"
set object 19 rect from 0.120552, 0 to 136.919140, 10 fc rgb "#FF00FF"
set object 20 rect from 0.121607, 0 to 137.397014, 10 fc rgb "#808080"
set object 21 rect from 0.122031, 0 to 138.011184, 10 fc rgb "#800080"
set object 22 rect from 0.122828, 0 to 138.616421, 10 fc rgb "#008080"
set object 23 rect from 0.123114, 0 to 154.891229, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.136449, 10 to 157.338976, 20 fc rgb "#FF0000"
set object 25 rect from 0.102546, 10 to 117.604783, 20 fc rgb "#00FF00"
set object 26 rect from 0.104714, 10 to 118.692327, 20 fc rgb "#0000FF"
set object 27 rect from 0.105428, 10 to 119.538717, 20 fc rgb "#FFFF00"
set object 28 rect from 0.106224, 10 to 120.848271, 20 fc rgb "#FF00FF"
set object 29 rect from 0.107360, 10 to 121.386938, 20 fc rgb "#808080"
set object 30 rect from 0.107839, 10 to 122.037248, 20 fc rgb "#800080"
set object 31 rect from 0.108653, 10 to 122.678489, 20 fc rgb "#008080"
set object 32 rect from 0.108975, 10 to 153.080155, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.134847, 20 to 155.187734, 30 fc rgb "#FF0000"
set object 34 rect from 0.083779, 20 to 96.377920, 30 fc rgb "#00FF00"
set object 35 rect from 0.085887, 20 to 97.191597, 30 fc rgb "#0000FF"
set object 36 rect from 0.086370, 20 to 98.127059, 30 fc rgb "#FFFF00"
set object 37 rect from 0.087201, 20 to 99.334039, 30 fc rgb "#FF00FF"
set object 38 rect from 0.088255, 20 to 99.803987, 30 fc rgb "#808080"
set object 39 rect from 0.088682, 20 to 100.446370, 30 fc rgb "#800080"
set object 40 rect from 0.089515, 20 to 101.152972, 30 fc rgb "#008080"
set object 41 rect from 0.089882, 20 to 151.357012, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133251, 30 to 153.566964, 40 fc rgb "#FF0000"
set object 43 rect from 0.067045, 30 to 77.445648, 40 fc rgb "#00FF00"
set object 44 rect from 0.069221, 30 to 78.330327, 40 fc rgb "#0000FF"
set object 45 rect from 0.069617, 30 to 79.493375, 40 fc rgb "#FFFF00"
set object 46 rect from 0.070662, 30 to 80.690211, 40 fc rgb "#FF00FF"
set object 47 rect from 0.071709, 30 to 81.191731, 40 fc rgb "#808080"
set object 48 rect from 0.072174, 30 to 81.820612, 40 fc rgb "#800080"
set object 49 rect from 0.072992, 30 to 82.519354, 40 fc rgb "#008080"
set object 50 rect from 0.073340, 30 to 149.515509, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.131643, 40 to 151.759316, 50 fc rgb "#FF0000"
set object 52 rect from 0.051097, 40 to 59.985155, 50 fc rgb "#00FF00"
set object 53 rect from 0.053581, 40 to 60.756041, 50 fc rgb "#0000FF"
set object 54 rect from 0.054020, 40 to 61.773725, 50 fc rgb "#FFFF00"
set object 55 rect from 0.054922, 40 to 63.074277, 50 fc rgb "#FF00FF"
set object 56 rect from 0.056103, 40 to 63.556586, 50 fc rgb "#808080"
set object 57 rect from 0.056511, 40 to 64.189968, 50 fc rgb "#800080"
set object 58 rect from 0.057352, 40 to 64.911280, 50 fc rgb "#008080"
set object 59 rect from 0.057747, 40 to 147.701076, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.129891, 50 to 152.762289, 60 fc rgb "#FF0000"
set object 61 rect from 0.029746, 50 to 36.885284, 60 fc rgb "#00FF00"
set object 62 rect from 0.033173, 50 to 38.171193, 60 fc rgb "#0000FF"
set object 63 rect from 0.034011, 50 to 40.741801, 60 fc rgb "#FFFF00"
set object 64 rect from 0.036343, 50 to 42.830168, 60 fc rgb "#FF00FF"
set object 65 rect from 0.038134, 50 to 43.519841, 60 fc rgb "#808080"
set object 66 rect from 0.038753, 50 to 44.289586, 60 fc rgb "#800080"
set object 67 rect from 0.039898, 50 to 45.537139, 60 fc rgb "#008080"
set object 68 rect from 0.040539, 50 to 145.218265, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
