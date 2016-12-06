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

set object 15 rect from 0.127763, 0 to 160.094257, 10 fc rgb "#FF0000"
set object 16 rect from 0.107107, 0 to 133.687751, 10 fc rgb "#00FF00"
set object 17 rect from 0.109286, 0 to 134.508851, 10 fc rgb "#0000FF"
set object 18 rect from 0.109707, 0 to 135.398717, 10 fc rgb "#FFFF00"
set object 19 rect from 0.110445, 0 to 136.689915, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111488, 0 to 137.207858, 10 fc rgb "#808080"
set object 21 rect from 0.111910, 0 to 137.859529, 10 fc rgb "#800080"
set object 22 rect from 0.112699, 0 to 138.561604, 10 fc rgb "#008080"
set object 23 rect from 0.113013, 0 to 156.175305, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.126196, 10 to 158.589444, 20 fc rgb "#FF0000"
set object 25 rect from 0.092071, 10 to 115.257925, 20 fc rgb "#00FF00"
set object 26 rect from 0.094283, 10 to 116.120723, 20 fc rgb "#0000FF"
set object 27 rect from 0.094727, 10 to 117.092816, 20 fc rgb "#FFFF00"
set object 28 rect from 0.095554, 10 to 118.545981, 20 fc rgb "#FF00FF"
set object 29 rect from 0.096745, 10 to 119.227134, 20 fc rgb "#808080"
set object 30 rect from 0.097274, 10 to 119.966007, 20 fc rgb "#800080"
set object 31 rect from 0.098127, 10 to 120.718413, 20 fc rgb "#008080"
set object 32 rect from 0.098476, 10 to 154.150136, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.124554, 20 to 156.231709, 30 fc rgb "#FF0000"
set object 34 rect from 0.077772, 20 to 97.663829, 30 fc rgb "#00FF00"
set object 35 rect from 0.079941, 20 to 98.386754, 30 fc rgb "#0000FF"
set object 36 rect from 0.080272, 20 to 99.404204, 30 fc rgb "#FFFF00"
set object 37 rect from 0.081108, 20 to 100.706448, 30 fc rgb "#FF00FF"
set object 38 rect from 0.082167, 20 to 101.212101, 30 fc rgb "#808080"
set object 39 rect from 0.082583, 20 to 101.903130, 30 fc rgb "#800080"
set object 40 rect from 0.083399, 20 to 102.656707, 30 fc rgb "#008080"
set object 41 rect from 0.083759, 20 to 152.231774, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.122958, 30 to 154.441076, 40 fc rgb "#FF0000"
set object 43 rect from 0.062585, 30 to 79.124781, 40 fc rgb "#00FF00"
set object 44 rect from 0.064904, 30 to 80.001040, 40 fc rgb "#0000FF"
set object 45 rect from 0.065314, 30 to 81.049216, 40 fc rgb "#FFFF00"
set object 46 rect from 0.066147, 30 to 82.412838, 40 fc rgb "#FF00FF"
set object 47 rect from 0.067260, 30 to 82.914833, 40 fc rgb "#808080"
set object 48 rect from 0.067684, 30 to 83.643829, 40 fc rgb "#800080"
set object 49 rect from 0.068545, 30 to 84.413427, 40 fc rgb "#008080"
set object 50 rect from 0.068910, 30 to 150.173538, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.121263, 40 to 152.544807, 50 fc rgb "#FF0000"
set object 52 rect from 0.047389, 40 to 60.817637, 50 fc rgb "#00FF00"
set object 53 rect from 0.049937, 40 to 61.675534, 50 fc rgb "#0000FF"
set object 54 rect from 0.050376, 40 to 62.719978, 50 fc rgb "#FFFF00"
set object 55 rect from 0.051236, 40 to 64.218573, 50 fc rgb "#FF00FF"
set object 56 rect from 0.052450, 40 to 64.797893, 50 fc rgb "#808080"
set object 57 rect from 0.052915, 40 to 65.486435, 50 fc rgb "#800080"
set object 58 rect from 0.053760, 40 to 66.285441, 50 fc rgb "#008080"
set object 59 rect from 0.054165, 40 to 148.074773, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.119436, 50 to 153.551356, 60 fc rgb "#FF0000"
set object 61 rect from 0.026229, 50 to 35.861882, 60 fc rgb "#00FF00"
set object 62 rect from 0.029682, 50 to 37.307732, 60 fc rgb "#0000FF"
set object 63 rect from 0.030525, 50 to 40.064385, 60 fc rgb "#FFFF00"
set object 64 rect from 0.032828, 50 to 42.185242, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034503, 50 to 43.022290, 60 fc rgb "#808080"
set object 66 rect from 0.035193, 50 to 43.923130, 60 fc rgb "#800080"
set object 67 rect from 0.036350, 50 to 45.195893, 60 fc rgb "#008080"
set object 68 rect from 0.037097, 50 to 145.336555, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0