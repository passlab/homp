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

set object 15 rect from 0.760025, 0 to 152.109572, 10 fc rgb "#FF0000"
set object 16 rect from 0.220355, 0 to 42.913343, 10 fc rgb "#00FF00"
set object 17 rect from 0.223905, 0 to 43.348388, 10 fc rgb "#0000FF"
set object 18 rect from 0.225900, 0 to 43.743499, 10 fc rgb "#FFFF00"
set object 19 rect from 0.228056, 0 to 44.131123, 10 fc rgb "#FF00FF"
set object 20 rect from 0.230013, 0 to 49.142008, 10 fc rgb "#808080"
set object 21 rect from 0.256326, 0 to 49.315949, 10 fc rgb "#800080"
set object 22 rect from 0.257250, 0 to 49.430758, 10 fc rgb "#008080"
set object 23 rect from 0.257612, 0 to 145.152696, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.868416, 10 to 173.695163, 20 fc rgb "#FF0000"
set object 25 rect from 0.679165, 10 to 131.047148, 20 fc rgb "#00FF00"
set object 26 rect from 0.683165, 10 to 131.350105, 20 fc rgb "#0000FF"
set object 27 rect from 0.684280, 10 to 131.753663, 20 fc rgb "#FFFF00"
set object 28 rect from 0.686582, 10 to 132.292958, 20 fc rgb "#FF00FF"
set object 29 rect from 0.689202, 10 to 138.050677, 20 fc rgb "#808080"
set object 30 rect from 0.719239, 10 to 138.252072, 20 fc rgb "#800080"
set object 31 rect from 0.720623, 10 to 138.385888, 20 fc rgb "#008080"
set object 32 rect from 0.720932, 10 to 166.507518, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.848769, 20 to 176.732796, 30 fc rgb "#FF0000"
set object 34 rect from 0.424535, 20 to 82.558282, 30 fc rgb "#00FF00"
set object 35 rect from 0.430390, 20 to 82.860856, 30 fc rgb "#0000FF"
set object 36 rect from 0.431708, 20 to 84.913975, 30 fc rgb "#FFFF00"
set object 37 rect from 0.442967, 20 to 91.274728, 30 fc rgb "#FF00FF"
set object 38 rect from 0.475565, 20 to 96.074043, 30 fc rgb "#808080"
set object 39 rect from 0.500619, 20 to 96.602010, 30 fc rgb "#800080"
set object 40 rect from 0.503883, 20 to 96.866954, 30 fc rgb "#008080"
set object 41 rect from 0.504678, 20 to 162.497273, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.775388, 30 to 163.262153, 40 fc rgb "#FF0000"
set object 43 rect from 0.286148, 30 to 55.504682, 40 fc rgb "#00FF00"
set object 44 rect from 0.289712, 30 to 55.915728, 40 fc rgb "#0000FF"
set object 45 rect from 0.291378, 30 to 57.177089, 40 fc rgb "#FFFF00"
set object 46 rect from 0.298010, 30 to 64.388157, 40 fc rgb "#FF00FF"
set object 47 rect from 0.335521, 30 to 69.766124, 40 fc rgb "#808080"
set object 48 rect from 0.363968, 30 to 70.193105, 40 fc rgb "#800080"
set object 49 rect from 0.366768, 30 to 70.621430, 40 fc rgb "#008080"
set object 50 rect from 0.367963, 30 to 148.117758, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.864047, 40 to 179.918261, 50 fc rgb "#FF0000"
set object 52 rect from 0.558644, 40 to 107.865368, 50 fc rgb "#00FF00"
set object 53 rect from 0.562273, 40 to 108.154118, 50 fc rgb "#0000FF"
set object 54 rect from 0.563468, 40 to 109.788512, 50 fc rgb "#FFFF00"
set object 55 rect from 0.572024, 40 to 116.599093, 50 fc rgb "#FF00FF"
set object 56 rect from 0.607440, 40 to 121.447173, 50 fc rgb "#808080"
set object 57 rect from 0.632794, 40 to 122.082078, 50 fc rgb "#800080"
set object 58 rect from 0.636629, 40 to 122.337038, 50 fc rgb "#008080"
set object 59 rect from 0.637335, 40 to 165.549114, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.766466, 50 to 160.609263, 60 fc rgb "#FF0000"
set object 61 rect from 0.101253, 50 to 20.850854, 60 fc rgb "#00FF00"
set object 62 rect from 0.109630, 50 to 21.636468, 60 fc rgb "#0000FF"
set object 63 rect from 0.112845, 50 to 23.961827, 60 fc rgb "#FFFF00"
set object 64 rect from 0.124992, 50 to 29.694587, 60 fc rgb "#FF00FF"
set object 65 rect from 0.154829, 50 to 34.282524, 60 fc rgb "#808080"
set object 66 rect from 0.179218, 50 to 34.699137, 60 fc rgb "#800080"
set object 67 rect from 0.182014, 50 to 35.350745, 60 fc rgb "#008080"
set object 68 rect from 0.184267, 50 to 146.645018, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
