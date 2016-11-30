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

set object 15 rect from 0.233234, 0 to 155.831304, 10 fc rgb "#FF0000"
set object 16 rect from 0.040939, 0 to 28.137508, 10 fc rgb "#00FF00"
set object 17 rect from 0.045653, 0 to 29.302460, 10 fc rgb "#0000FF"
set object 18 rect from 0.047063, 0 to 30.801602, 10 fc rgb "#FFFF00"
set object 19 rect from 0.049481, 0 to 31.956557, 10 fc rgb "#FF00FF"
set object 20 rect from 0.051310, 0 to 38.534007, 10 fc rgb "#808080"
set object 21 rect from 0.061887, 0 to 38.994373, 10 fc rgb "#800080"
set object 22 rect from 0.063016, 0 to 39.517195, 10 fc rgb "#008080"
set object 23 rect from 0.063417, 0 to 144.951328, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.238321, 10 to 156.860706, 20 fc rgb "#FF0000"
set object 25 rect from 0.109175, 10 to 69.212549, 20 fc rgb "#00FF00"
set object 26 rect from 0.111167, 10 to 69.611698, 20 fc rgb "#0000FF"
set object 27 rect from 0.111563, 10 to 70.077677, 20 fc rgb "#FFFF00"
set object 28 rect from 0.112352, 10 to 70.797882, 20 fc rgb "#FF00FF"
set object 29 rect from 0.113486, 10 to 77.282895, 20 fc rgb "#808080"
set object 30 rect from 0.123916, 10 to 77.747627, 20 fc rgb "#800080"
set object 31 rect from 0.124939, 10 to 78.207360, 20 fc rgb "#008080"
set object 32 rect from 0.125353, 10 to 148.366230, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.243367, 20 to 163.509361, 30 fc rgb "#FF0000"
set object 34 rect from 0.170449, 20 to 107.366777, 30 fc rgb "#00FF00"
set object 35 rect from 0.172272, 20 to 107.753426, 30 fc rgb "#0000FF"
set object 36 rect from 0.172624, 20 to 109.830976, 30 fc rgb "#FFFF00"
set object 37 rect from 0.175950, 20 to 113.334582, 30 fc rgb "#FF00FF"
set object 38 rect from 0.181563, 20 to 118.898872, 30 fc rgb "#808080"
set object 39 rect from 0.190470, 20 to 119.327368, 30 fc rgb "#800080"
set object 40 rect from 0.191462, 20 to 119.755874, 30 fc rgb "#008080"
set object 41 rect from 0.191860, 20 to 151.413844, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.236151, 30 to 160.216252, 40 fc rgb "#FF0000"
set object 43 rect from 0.075207, 30 to 48.274637, 40 fc rgb "#00FF00"
set object 44 rect from 0.077652, 30 to 48.868041, 40 fc rgb "#0000FF"
set object 45 rect from 0.078356, 30 to 50.997436, 40 fc rgb "#FFFF00"
set object 46 rect from 0.081831, 30 to 55.393032, 40 fc rgb "#FF00FF"
set object 47 rect from 0.088836, 30 to 61.014779, 40 fc rgb "#808080"
set object 48 rect from 0.097816, 30 to 61.512619, 40 fc rgb "#800080"
set object 49 rect from 0.098934, 30 to 62.320280, 40 fc rgb "#008080"
set object 50 rect from 0.099907, 30 to 146.850232, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.240832, 40 to 163.142074, 50 fc rgb "#FF0000"
set object 52 rect from 0.137381, 40 to 87.310226, 50 fc rgb "#00FF00"
set object 53 rect from 0.140215, 40 to 87.868027, 50 fc rgb "#0000FF"
set object 54 rect from 0.140850, 40 to 90.711390, 50 fc rgb "#FFFF00"
set object 55 rect from 0.145369, 40 to 94.472972, 50 fc rgb "#FF00FF"
set object 56 rect from 0.151403, 40 to 100.111586, 50 fc rgb "#808080"
set object 57 rect from 0.160411, 40 to 100.608802, 50 fc rgb "#800080"
set object 58 rect from 0.161491, 40 to 101.127873, 50 fc rgb "#008080"
set object 59 rect from 0.162019, 40 to 149.815397, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.244955, 50 to 164.675579, 60 fc rgb "#FF0000"
set object 61 rect from 0.201437, 50 to 126.746211, 60 fc rgb "#00FF00"
set object 62 rect from 0.203278, 50 to 127.139737, 60 fc rgb "#0000FF"
set object 63 rect from 0.203663, 50 to 129.192305, 60 fc rgb "#FFFF00"
set object 64 rect from 0.206985, 50 to 132.888919, 60 fc rgb "#FF00FF"
set object 65 rect from 0.212904, 50 to 138.500679, 60 fc rgb "#808080"
set object 66 rect from 0.221851, 50 to 138.917940, 60 fc rgb "#800080"
set object 67 rect from 0.222784, 50 to 139.315833, 60 fc rgb "#008080"
set object 68 rect from 0.223156, 50 to 152.692483, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
