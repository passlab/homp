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

set object 15 rect from 0.233608, 0 to 156.439237, 10 fc rgb "#FF0000"
set object 16 rect from 0.034593, 0 to 23.382369, 10 fc rgb "#00FF00"
set object 17 rect from 0.037988, 0 to 24.192949, 10 fc rgb "#0000FF"
set object 18 rect from 0.039003, 0 to 25.610062, 10 fc rgb "#FFFF00"
set object 19 rect from 0.041309, 0 to 26.438061, 10 fc rgb "#FF00FF"
set object 20 rect from 0.042631, 0 to 34.648984, 10 fc rgb "#808080"
set object 21 rect from 0.055827, 0 to 34.996727, 10 fc rgb "#800080"
set object 22 rect from 0.056744, 0 to 35.431569, 10 fc rgb "#008080"
set object 23 rect from 0.057072, 0 to 144.640127, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.241501, 10 to 159.187626, 20 fc rgb "#FF0000"
set object 25 rect from 0.210217, 10 to 131.580660, 20 fc rgb "#00FF00"
set object 26 rect from 0.211818, 10 to 131.970711, 20 fc rgb "#0000FF"
set object 27 rect from 0.212243, 10 to 132.374444, 20 fc rgb "#FFFF00"
set object 28 rect from 0.212922, 10 to 132.982224, 20 fc rgb "#FF00FF"
set object 29 rect from 0.213884, 10 to 140.644466, 20 fc rgb "#808080"
set object 30 rect from 0.226225, 10 to 140.963594, 20 fc rgb "#800080"
set object 31 rect from 0.226951, 10 to 141.289567, 20 fc rgb "#008080"
set object 32 rect from 0.227228, 10 to 149.945283, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.235761, 20 to 162.572397, 30 fc rgb "#FF0000"
set object 34 rect from 0.066155, 20 to 42.199868, 30 fc rgb "#00FF00"
set object 35 rect from 0.068150, 20 to 42.574990, 30 fc rgb "#0000FF"
set object 36 rect from 0.068541, 20 to 44.662709, 30 fc rgb "#FFFF00"
set object 37 rect from 0.071922, 20 to 50.332410, 30 fc rgb "#FF00FF"
set object 38 rect from 0.081029, 20 to 57.973499, 30 fc rgb "#808080"
set object 39 rect from 0.093332, 20 to 58.546443, 30 fc rgb "#800080"
set object 40 rect from 0.094460, 20 to 59.113784, 30 fc rgb "#008080"
set object 41 rect from 0.095158, 20 to 146.062839, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.238860, 30 to 163.682818, 40 fc rgb "#FF0000"
set object 43 rect from 0.143079, 30 to 89.757777, 40 fc rgb "#00FF00"
set object 44 rect from 0.144595, 30 to 90.172708, 40 fc rgb "#0000FF"
set object 45 rect from 0.145052, 30 to 91.888420, 40 fc rgb "#FFFF00"
set object 46 rect from 0.147819, 30 to 97.107729, 40 fc rgb "#FF00FF"
set object 47 rect from 0.156211, 30 to 104.733268, 40 fc rgb "#808080"
set object 48 rect from 0.168471, 30 to 105.248354, 40 fc rgb "#800080"
set object 49 rect from 0.169531, 30 to 105.606053, 40 fc rgb "#008080"
set object 50 rect from 0.169864, 30 to 148.235165, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.240200, 40 to 164.617823, 50 fc rgb "#FF0000"
set object 52 rect from 0.176510, 40 to 110.501878, 50 fc rgb "#00FF00"
set object 53 rect from 0.177936, 40 to 110.796125, 50 fc rgb "#0000FF"
set object 54 rect from 0.178203, 40 to 112.504996, 50 fc rgb "#FFFF00"
set object 55 rect from 0.180956, 40 to 117.912797, 50 fc rgb "#FF00FF"
set object 56 rect from 0.189654, 40 to 125.618585, 50 fc rgb "#808080"
set object 57 rect from 0.202046, 40 to 126.088880, 50 fc rgb "#800080"
set object 58 rect from 0.203016, 40 to 126.416099, 50 fc rgb "#008080"
set object 59 rect from 0.203319, 40 to 149.155857, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.237368, 50 to 162.760889, 60 fc rgb "#FF0000"
set object 61 rect from 0.108422, 50 to 68.244152, 60 fc rgb "#00FF00"
set object 62 rect from 0.110010, 50 to 68.670281, 60 fc rgb "#0000FF"
set object 63 rect from 0.110486, 50 to 70.269045, 60 fc rgb "#FFFF00"
set object 64 rect from 0.113066, 50 to 75.587884, 60 fc rgb "#FF00FF"
set object 65 rect from 0.121618, 50 to 83.239550, 60 fc rgb "#808080"
set object 66 rect from 0.133919, 50 to 83.739707, 60 fc rgb "#800080"
set object 67 rect from 0.134946, 50 to 84.126022, 60 fc rgb "#008080"
set object 68 rect from 0.135333, 50 to 147.275911, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
