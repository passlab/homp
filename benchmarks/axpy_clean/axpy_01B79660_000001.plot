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

set object 15 rect from 0.924109, 0 to 153.942186, 10 fc rgb "#FF0000"
set object 16 rect from 0.357748, 0 to 59.469754, 10 fc rgb "#00FF00"
set object 17 rect from 0.373673, 0 to 60.215772, 10 fc rgb "#0000FF"
set object 18 rect from 0.376526, 0 to 60.992084, 10 fc rgb "#FFFF00"
set object 19 rect from 0.381441, 0 to 62.196707, 10 fc rgb "#FF00FF"
set object 20 rect from 0.388843, 0 to 65.544646, 10 fc rgb "#808080"
set object 21 rect from 0.409925, 0 to 66.169639, 10 fc rgb "#800080"
set object 22 rect from 0.415569, 0 to 66.787100, 10 fc rgb "#008080"
set object 23 rect from 0.417541, 0 to 147.457903, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.912594, 10 to 153.394142, 20 fc rgb "#FF0000"
set object 25 rect from 0.116580, 10 to 21.439567, 20 fc rgb "#00FF00"
set object 26 rect from 0.136912, 10 to 22.485499, 20 fc rgb "#0000FF"
set object 27 rect from 0.141126, 10 to 23.778285, 20 fc rgb "#FFFF00"
set object 28 rect from 0.149306, 10 to 25.318887, 20 fc rgb "#FF00FF"
set object 29 rect from 0.158808, 10 to 28.830809, 20 fc rgb "#808080"
set object 30 rect from 0.180757, 10 to 29.484015, 20 fc rgb "#800080"
set object 31 rect from 0.187605, 10 to 30.310820, 20 fc rgb "#008080"
set object 32 rect from 0.190048, 10 to 145.305610, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.945078, 20 to 158.397292, 30 fc rgb "#FF0000"
set object 34 rect from 0.701567, 20 to 113.833127, 30 fc rgb "#00FF00"
set object 35 rect from 0.712510, 20 to 114.469821, 30 fc rgb "#0000FF"
set object 36 rect from 0.714877, 20 to 116.343360, 30 fc rgb "#FFFF00"
set object 37 rect from 0.726602, 20 to 117.655059, 30 fc rgb "#FF00FF"
set object 38 rect from 0.734740, 20 to 120.851838, 30 fc rgb "#808080"
set object 39 rect from 0.754780, 20 to 121.497991, 30 fc rgb "#800080"
set object 40 rect from 0.760434, 20 to 122.151198, 30 fc rgb "#008080"
set object 41 rect from 0.763002, 20 to 150.899614, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.954727, 30 to 160.163272, 40 fc rgb "#FF0000"
set object 43 rect from 0.597244, 30 to 97.295244, 40 fc rgb "#00FF00"
set object 44 rect from 0.609595, 30 to 97.947488, 40 fc rgb "#0000FF"
set object 45 rect from 0.611860, 30 to 99.934836, 40 fc rgb "#FFFF00"
set object 46 rect from 0.624427, 30 to 101.439853, 40 fc rgb "#FF00FF"
set object 47 rect from 0.633592, 30 to 104.661319, 40 fc rgb "#808080"
set object 48 rect from 0.653688, 30 to 105.253612, 40 fc rgb "#800080"
set object 49 rect from 0.659036, 30 to 105.976866, 40 fc rgb "#008080"
set object 50 rect from 0.661884, 30 to 152.518121, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.964012, 40 to 161.460064, 50 fc rgb "#FF0000"
set object 52 rect from 0.806631, 40 to 130.665793, 50 fc rgb "#00FF00"
set object 53 rect from 0.817492, 40 to 131.264816, 50 fc rgb "#0000FF"
set object 54 rect from 0.819639, 40 to 133.250722, 50 fc rgb "#FFFF00"
set object 55 rect from 0.832289, 40 to 134.582941, 50 fc rgb "#FF00FF"
set object 56 rect from 0.840338, 40 to 137.781004, 50 fc rgb "#808080"
set object 57 rect from 0.860304, 40 to 138.375219, 50 fc rgb "#800080"
set object 58 rect from 0.865673, 40 to 139.008068, 50 fc rgb "#008080"
set object 59 rect from 0.867971, 40 to 154.007911, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.934946, 50 to 157.616006, 60 fc rgb "#FF0000"
set object 61 rect from 0.473893, 50 to 77.772493, 60 fc rgb "#00FF00"
set object 62 rect from 0.487678, 50 to 78.489656, 60 fc rgb "#0000FF"
set object 63 rect from 0.490396, 50 to 80.679459, 60 fc rgb "#FFFF00"
set object 64 rect from 0.504441, 50 to 82.410653, 60 fc rgb "#FF00FF"
set object 65 rect from 0.514946, 50 to 85.639812, 60 fc rgb "#808080"
set object 66 rect from 0.535100, 50 to 86.362586, 60 fc rgb "#800080"
set object 67 rect from 0.541427, 50 to 87.404990, 60 fc rgb "#008080"
set object 68 rect from 0.546119, 50 to 149.142612, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
