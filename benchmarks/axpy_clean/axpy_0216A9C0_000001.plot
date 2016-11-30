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

set object 15 rect from 0.143841, 0 to 157.535336, 10 fc rgb "#FF0000"
set object 16 rect from 0.107236, 0 to 116.564426, 10 fc rgb "#00FF00"
set object 17 rect from 0.109589, 0 to 117.383978, 10 fc rgb "#0000FF"
set object 18 rect from 0.110112, 0 to 118.186453, 10 fc rgb "#FFFF00"
set object 19 rect from 0.110890, 0 to 119.353862, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111986, 0 to 120.778458, 10 fc rgb "#808080"
set object 21 rect from 0.113322, 0 to 121.426213, 10 fc rgb "#800080"
set object 22 rect from 0.114188, 0 to 122.060070, 10 fc rgb "#008080"
set object 23 rect from 0.114500, 0 to 152.806929, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138897, 10 to 152.889170, 20 fc rgb "#FF0000"
set object 25 rect from 0.057711, 10 to 64.032357, 20 fc rgb "#00FF00"
set object 26 rect from 0.060364, 10 to 65.070680, 20 fc rgb "#0000FF"
set object 27 rect from 0.061092, 10 to 65.959594, 20 fc rgb "#FFFF00"
set object 28 rect from 0.061980, 10 to 67.270019, 20 fc rgb "#FF00FF"
set object 29 rect from 0.063188, 10 to 68.905880, 20 fc rgb "#808080"
set object 30 rect from 0.064736, 10 to 69.580317, 20 fc rgb "#800080"
set object 31 rect from 0.065612, 10 to 70.249379, 20 fc rgb "#008080"
set object 32 rect from 0.065968, 10 to 147.447876, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.136934, 20 to 154.299805, 30 fc rgb "#FF0000"
set object 34 rect from 0.033274, 20 to 38.819593, 30 fc rgb "#00FF00"
set object 35 rect from 0.036842, 20 to 40.629414, 30 fc rgb "#0000FF"
set object 36 rect from 0.038227, 20 to 43.060307, 30 fc rgb "#FFFF00"
set object 37 rect from 0.040510, 20 to 45.378079, 30 fc rgb "#FF00FF"
set object 38 rect from 0.042653, 20 to 47.124963, 30 fc rgb "#808080"
set object 39 rect from 0.044430, 20 to 48.107790, 30 fc rgb "#800080"
set object 40 rect from 0.046067, 20 to 49.656162, 30 fc rgb "#008080"
set object 41 rect from 0.046690, 20 to 145.114076, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.142134, 30 to 155.512088, 40 fc rgb "#FF0000"
set object 43 rect from 0.091884, 30 to 99.973944, 40 fc rgb "#00FF00"
set object 44 rect from 0.094047, 30 to 100.614225, 40 fc rgb "#0000FF"
set object 45 rect from 0.094397, 30 to 101.537263, 40 fc rgb "#FFFF00"
set object 46 rect from 0.095269, 30 to 102.775147, 40 fc rgb "#FF00FF"
set object 47 rect from 0.096425, 30 to 104.006574, 40 fc rgb "#808080"
set object 48 rect from 0.097582, 30 to 104.571102, 40 fc rgb "#800080"
set object 49 rect from 0.098364, 30 to 105.234853, 40 fc rgb "#008080"
set object 50 rect from 0.098733, 30 to 151.109148, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.145349, 40 to 159.240590, 50 fc rgb "#FF0000"
set object 52 rect from 0.123193, 40 to 133.332048, 50 fc rgb "#00FF00"
set object 53 rect from 0.125307, 40 to 134.092893, 50 fc rgb "#0000FF"
set object 54 rect from 0.125773, 40 to 135.017012, 50 fc rgb "#FFFF00"
set object 55 rect from 0.126653, 40 to 136.331699, 50 fc rgb "#FF00FF"
set object 56 rect from 0.127871, 40 to 137.588760, 50 fc rgb "#808080"
set object 57 rect from 0.129050, 40 to 138.233302, 50 fc rgb "#800080"
set object 58 rect from 0.129908, 40 to 138.939733, 50 fc rgb "#008080"
set object 59 rect from 0.130338, 40 to 154.573021, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.140604, 50 to 154.736295, 60 fc rgb "#FF0000"
set object 61 rect from 0.074580, 50 to 81.691024, 60 fc rgb "#00FF00"
set object 62 rect from 0.076993, 50 to 82.779499, 60 fc rgb "#0000FF"
set object 63 rect from 0.077686, 50 to 83.824182, 60 fc rgb "#FFFF00"
set object 64 rect from 0.078682, 50 to 85.245598, 60 fc rgb "#FF00FF"
set object 65 rect from 0.080014, 50 to 86.523998, 60 fc rgb "#808080"
set object 66 rect from 0.081215, 50 to 87.275239, 60 fc rgb "#800080"
set object 67 rect from 0.082168, 50 to 88.062798, 60 fc rgb "#008080"
set object 68 rect from 0.082639, 50 to 149.415629, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
