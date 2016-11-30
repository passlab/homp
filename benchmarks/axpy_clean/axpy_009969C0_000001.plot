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

set object 15 rect from 0.175419, 0 to 155.943242, 10 fc rgb "#FF0000"
set object 16 rect from 0.119908, 0 to 106.046067, 10 fc rgb "#00FF00"
set object 17 rect from 0.122281, 0 to 106.714992, 10 fc rgb "#0000FF"
set object 18 rect from 0.122793, 0 to 107.356953, 10 fc rgb "#FFFF00"
set object 19 rect from 0.123543, 0 to 108.312934, 10 fc rgb "#FF00FF"
set object 20 rect from 0.124656, 0 to 109.537703, 10 fc rgb "#808080"
set object 21 rect from 0.126062, 0 to 110.071800, 10 fc rgb "#800080"
set object 22 rect from 0.126930, 0 to 110.597198, 10 fc rgb "#008080"
set object 23 rect from 0.127276, 0 to 151.936645, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.171046, 10 to 152.910023, 20 fc rgb "#FF0000"
set object 25 rect from 0.081996, 10 to 73.132128, 20 fc rgb "#00FF00"
set object 26 rect from 0.084531, 10 to 74.178576, 20 fc rgb "#0000FF"
set object 27 rect from 0.085390, 10 to 74.949276, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086326, 10 to 76.124463, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087677, 10 to 77.553651, 20 fc rgb "#808080"
set object 30 rect from 0.089293, 10 to 78.059912, 20 fc rgb "#800080"
set object 31 rect from 0.090228, 10 to 78.685344, 20 fc rgb "#008080"
set object 32 rect from 0.090587, 10 to 147.943097, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.177103, 20 to 157.725592, 30 fc rgb "#FF0000"
set object 34 rect from 0.135719, 20 to 119.515920, 30 fc rgb "#00FF00"
set object 35 rect from 0.137753, 20 to 120.170928, 30 fc rgb "#0000FF"
set object 36 rect from 0.138265, 20 to 120.906834, 30 fc rgb "#FFFF00"
set object 37 rect from 0.139116, 20 to 122.243816, 30 fc rgb "#FF00FF"
set object 38 rect from 0.140661, 20 to 123.278085, 30 fc rgb "#808080"
set object 39 rect from 0.141859, 20 to 123.836538, 30 fc rgb "#800080"
set object 40 rect from 0.142747, 20 to 124.400210, 30 fc rgb "#008080"
set object 41 rect from 0.143146, 20 to 153.548504, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.178601, 30 to 158.443232, 40 fc rgb "#FF0000"
set object 43 rect from 0.152961, 30 to 134.561099, 40 fc rgb "#00FF00"
set object 44 rect from 0.155040, 30 to 135.135210, 40 fc rgb "#0000FF"
set object 45 rect from 0.155463, 30 to 135.836320, 40 fc rgb "#FFFF00"
set object 46 rect from 0.156269, 30 to 136.808829, 40 fc rgb "#FF00FF"
set object 47 rect from 0.157386, 30 to 137.796125, 40 fc rgb "#808080"
set object 48 rect from 0.158524, 30 to 138.238017, 40 fc rgb "#800080"
set object 49 rect from 0.159313, 30 to 138.772983, 40 fc rgb "#008080"
set object 50 rect from 0.159649, 30 to 154.941158, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.173140, 40 to 154.389662, 50 fc rgb "#FF0000"
set object 52 rect from 0.100472, 40 to 88.996720, 50 fc rgb "#00FF00"
set object 53 rect from 0.102741, 40 to 89.821351, 50 fc rgb "#0000FF"
set object 54 rect from 0.103388, 40 to 90.644244, 50 fc rgb "#FFFF00"
set object 55 rect from 0.104328, 40 to 91.915985, 50 fc rgb "#FF00FF"
set object 56 rect from 0.105810, 40 to 92.974611, 50 fc rgb "#808080"
set object 57 rect from 0.107021, 40 to 93.515667, 50 fc rgb "#800080"
set object 58 rect from 0.107932, 40 to 94.206339, 50 fc rgb "#008080"
set object 59 rect from 0.108413, 40 to 149.903771, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.168708, 50 to 153.979087, 60 fc rgb "#FF0000"
set object 61 rect from 0.055030, 50 to 50.914047, 60 fc rgb "#00FF00"
set object 62 rect from 0.059019, 50 to 52.646817, 60 fc rgb "#0000FF"
set object 63 rect from 0.060648, 50 to 54.965876, 60 fc rgb "#FFFF00"
set object 64 rect from 0.063356, 50 to 56.817818, 60 fc rgb "#FF00FF"
set object 65 rect from 0.065449, 50 to 58.263532, 60 fc rgb "#808080"
set object 66 rect from 0.067116, 50 to 58.937678, 60 fc rgb "#800080"
set object 67 rect from 0.068602, 50 to 60.297276, 60 fc rgb "#008080"
set object 68 rect from 0.069440, 50 to 145.430927, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
