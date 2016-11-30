set title "Offloading (matvec_kernel) Profile on 6 Devices"
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

set object 15 rect from 0.355207, 0 to 399.125618, 10 fc rgb "#FF0000"
set object 16 rect from -3.794360, 0 to -1585.505888, 10 fc rgb "#00FF00"
set object 17 rect from 0.283968, 0 to 120.388456, 10 fc rgb "#0000FF"
set object 18 rect from 0.284446, 0 to 123.044480, 10 fc rgb "#FFFF00"
set object 19 rect from 0.285449, 0 to 124.072101, 10 fc rgb "#FF00FF"
set object 20 rect from 0.286487, 0 to 357.287759, 10 fc rgb "#808080"
set object 21 rect from 0.338987, 0 to 143.781335, 10 fc rgb "#800080"
set object 22 rect from 0.339815, 0 to 142.884558, 10 fc rgb "#008080"
set object 23 rect from 0.340148, 0 to 235.824846, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.353680, 10 to 459.600809, 20 fc rgb "#FF0000"
set object 25 rect from -3.879190, 10 to -1620.596470, 20 fc rgb "#00FF00"
set object 26 rect from 0.226065, 10 to 97.660731, 20 fc rgb "#0000FF"
set object 27 rect from 0.226519, 10 to 101.761609, 20 fc rgb "#FFFF00"
set object 28 rect from 0.228002, 10 to 101.029572, 20 fc rgb "#FF00FF"
set object 29 rect from 0.229054, 10 to 388.371097, 20 fc rgb "#808080"
set object 30 rect from 0.281231, 10 to 120.281858, 20 fc rgb "#800080"
set object 31 rect from 0.282068, 10 to 118.828612, 20 fc rgb "#008080"
set object 32 rect from 0.282392, 10 to 481.342285, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.352128, 20 to 449.786088, 30 fc rgb "#FF0000"
set object 34 rect from -3.965577, 20 to -1656.559995, 30 fc rgb "#00FF00"
set object 35 rect from 0.169113, 20 to 73.329267, 30 fc rgb "#0000FF"
set object 36 rect from 0.169794, 20 to 79.630884, 30 fc rgb "#FFFF00"
set object 37 rect from 0.171139, 20 to 122.988468, 30 fc rgb "#FF00FF"
set object 38 rect from 0.177496, 20 to 309.594316, 30 fc rgb "#808080"
set object 39 rect from 0.222074, 20 to 96.798654, 30 fc rgb "#800080"
set object 40 rect from 0.222997, 20 to 97.646932, 30 fc rgb "#008080"
set object 41 rect from 0.223692, 20 to 797.317268, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.350366, 30 to 975.748970, 40 fc rgb "#FF0000"
set object 43 rect from -5.433215, 30 to -2270.776514, 40 fc rgb "#00FF00"
set object 44 rect from 0.112643, 30 to 49.001572, 40 fc rgb "#0000FF"
set object 45 rect from 0.113217, 30 to 53.899302, 40 fc rgb "#FFFF00"
set object 46 rect from 0.114480, 30 to 93.134234, 40 fc rgb "#FF00FF"
set object 47 rect from 0.121563, 30 to 822.650443, 40 fc rgb "#808080"
set object 48 rect from 0.165976, 30 to 72.559590, 40 fc rgb "#800080"
set object 49 rect from 0.166916, 30 to 86.894080, 40 fc rgb "#008080"
set object 50 rect from 0.167406, 30 to 1108.136963, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.348551, 40 to 387.492280, 50 fc rgb "#FF0000"
set object 52 rect from -5.509769, 40 to -2302.676776, 50 fc rgb "#00FF00"
set object 53 rect from 0.057934, 40 to 26.186032, 50 fc rgb "#0000FF"
set object 54 rect from 0.058418, 40 to 31.291133, 50 fc rgb "#FFFF00"
set object 55 rect from 0.059971, 40 to 63.368239, 50 fc rgb "#FF00FF"
set object 56 rect from 0.066411, 40 to 219.702241, 50 fc rgb "#808080"
set object 57 rect from 0.109336, 40 to 47.873608, 50 fc rgb "#800080"
set object 58 rect from 0.110198, 40 to 47.716829, 50 fc rgb "#008080"
set object 59 rect from 0.110837, 40 to 1940.290196, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.346643, 50 to 403.744061, 60 fc rgb "#FF0000"
set object 61 rect from -5.591475, 50 to -2336.326261, 60 fc rgb "#00FF00"
set object 62 rect from 0.001465, 50 to 3.137218, 60 fc rgb "#0000FF"
set object 63 rect from 0.002009, 50 to 8.940496, 60 fc rgb "#FFFF00"
set object 64 rect from 0.003123, 50 to 35.807575, 60 fc rgb "#FF00FF"
set object 65 rect from 0.011109, 50 to 215.097585, 60 fc rgb "#808080"
set object 66 rect from 0.054473, 50 to 25.466100, 60 fc rgb "#800080"
set object 67 rect from 0.055373, 50 to 26.471574, 60 fc rgb "#008080"
set object 68 rect from 0.055886, 50 to 2168.676684, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
