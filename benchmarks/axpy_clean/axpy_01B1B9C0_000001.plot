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

set object 15 rect from 0.144265, 0 to 156.253769, 10 fc rgb "#FF0000"
set object 16 rect from 0.093780, 0 to 101.131342, 10 fc rgb "#00FF00"
set object 17 rect from 0.096123, 0 to 101.878923, 10 fc rgb "#0000FF"
set object 18 rect from 0.096596, 0 to 102.687746, 10 fc rgb "#FFFF00"
set object 19 rect from 0.097376, 0 to 103.859799, 10 fc rgb "#FF00FF"
set object 20 rect from 0.098488, 0 to 105.243034, 10 fc rgb "#808080"
set object 21 rect from 0.099795, 0 to 105.812166, 10 fc rgb "#800080"
set object 22 rect from 0.100577, 0 to 106.399249, 10 fc rgb "#008080"
set object 23 rect from 0.100901, 0 to 151.663753, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.138724, 10 to 153.801958, 20 fc rgb "#FF0000"
set object 25 rect from 0.036776, 10 to 42.534992, 20 fc rgb "#00FF00"
set object 26 rect from 0.040767, 10 to 44.100897, 20 fc rgb "#0000FF"
set object 27 rect from 0.041906, 10 to 46.549539, 20 fc rgb "#FFFF00"
set object 28 rect from 0.044265, 10 to 48.142898, 20 fc rgb "#FF00FF"
set object 29 rect from 0.045736, 10 to 50.136446, 20 fc rgb "#808080"
set object 30 rect from 0.047642, 10 to 50.863964, 20 fc rgb "#800080"
set object 31 rect from 0.048741, 10 to 51.724526, 20 fc rgb "#008080"
set object 32 rect from 0.049120, 10 to 145.278700, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.140719, 20 to 153.485187, 30 fc rgb "#FF0000"
set object 34 rect from 0.059587, 20 to 65.289728, 30 fc rgb "#00FF00"
set object 35 rect from 0.062204, 20 to 66.053147, 30 fc rgb "#0000FF"
set object 36 rect from 0.062669, 20 to 67.243151, 30 fc rgb "#FFFF00"
set object 37 rect from 0.063827, 20 to 68.932598, 30 fc rgb "#FF00FF"
set object 38 rect from 0.065425, 20 to 70.226081, 30 fc rgb "#808080"
set object 39 rect from 0.066664, 20 to 71.027512, 30 fc rgb "#800080"
set object 40 rect from 0.067657, 20 to 72.179503, 30 fc rgb "#008080"
set object 41 rect from 0.068485, 20 to 147.772745, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.145832, 30 to 157.979116, 40 fc rgb "#FF0000"
set object 43 rect from 0.108355, 30 to 116.181146, 40 fc rgb "#00FF00"
set object 44 rect from 0.110395, 30 to 116.889657, 40 fc rgb "#0000FF"
set object 45 rect from 0.110814, 30 to 117.797735, 40 fc rgb "#FFFF00"
set object 46 rect from 0.111676, 30 to 119.038423, 40 fc rgb "#FF00FF"
set object 47 rect from 0.112849, 30 to 120.283334, 40 fc rgb "#808080"
set object 48 rect from 0.114030, 30 to 120.924268, 40 fc rgb "#800080"
set object 49 rect from 0.114892, 30 to 121.577873, 40 fc rgb "#008080"
set object 50 rect from 0.115257, 30 to 153.419721, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.142539, 40 to 154.938111, 50 fc rgb "#FF0000"
set object 52 rect from 0.077618, 40 to 83.947557, 50 fc rgb "#00FF00"
set object 53 rect from 0.079851, 40 to 84.855634, 50 fc rgb "#0000FF"
set object 54 rect from 0.080476, 40 to 85.900979, 50 fc rgb "#FFFF00"
set object 55 rect from 0.081466, 40 to 87.217692, 50 fc rgb "#FF00FF"
set object 56 rect from 0.082711, 40 to 88.434094, 50 fc rgb "#808080"
set object 57 rect from 0.083876, 40 to 89.113040, 50 fc rgb "#800080"
set object 58 rect from 0.084762, 40 to 89.816272, 50 fc rgb "#008080"
set object 59 rect from 0.085193, 40 to 149.783188, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.147337, 50 to 159.352848, 60 fc rgb "#FF0000"
set object 61 rect from 0.124132, 50 to 132.877104, 60 fc rgb "#00FF00"
set object 62 rect from 0.126205, 50 to 133.485305, 60 fc rgb "#0000FF"
set object 63 rect from 0.126549, 50 to 134.394438, 60 fc rgb "#FFFF00"
set object 64 rect from 0.127391, 50 to 135.555933, 60 fc rgb "#FF00FF"
set object 65 rect from 0.128491, 50 to 136.787117, 60 fc rgb "#808080"
set object 66 rect from 0.129672, 50 to 137.433331, 60 fc rgb "#800080"
set object 67 rect from 0.130517, 50 to 138.078488, 60 fc rgb "#008080"
set object 68 rect from 0.130884, 50 to 155.045814, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
