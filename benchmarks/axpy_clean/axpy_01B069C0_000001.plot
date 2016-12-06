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

set object 15 rect from 0.133602, 0 to 158.688265, 10 fc rgb "#FF0000"
set object 16 rect from 0.095555, 0 to 112.689772, 10 fc rgb "#00FF00"
set object 17 rect from 0.097876, 0 to 113.534681, 10 fc rgb "#0000FF"
set object 18 rect from 0.098361, 0 to 114.460500, 10 fc rgb "#FFFF00"
set object 19 rect from 0.099175, 0 to 115.734223, 10 fc rgb "#FF00FF"
set object 20 rect from 0.100264, 0 to 117.270320, 10 fc rgb "#808080"
set object 21 rect from 0.101585, 0 to 117.881752, 10 fc rgb "#800080"
set object 22 rect from 0.102392, 0 to 118.529016, 10 fc rgb "#008080"
set object 23 rect from 0.102670, 0 to 153.787555, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.128763, 10 to 153.374922, 20 fc rgb "#FF0000"
set object 25 rect from 0.048422, 10 to 58.603950, 20 fc rgb "#00FF00"
set object 26 rect from 0.051083, 10 to 59.497406, 20 fc rgb "#0000FF"
set object 27 rect from 0.051591, 10 to 60.420913, 20 fc rgb "#FFFF00"
set object 28 rect from 0.052452, 10 to 61.911933, 20 fc rgb "#FF00FF"
set object 29 rect from 0.053696, 10 to 63.405260, 20 fc rgb "#808080"
set object 30 rect from 0.054988, 10 to 64.081421, 20 fc rgb "#800080"
set object 31 rect from 0.055868, 10 to 64.848890, 20 fc rgb "#008080"
set object 32 rect from 0.056238, 10 to 147.998008, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.126813, 20 to 154.120432, 30 fc rgb "#FF0000"
set object 34 rect from 0.026580, 20 to 34.207886, 30 fc rgb "#00FF00"
set object 35 rect from 0.030184, 20 to 35.706996, 30 fc rgb "#0000FF"
set object 36 rect from 0.031023, 20 to 38.144636, 30 fc rgb "#FFFF00"
set object 37 rect from 0.033152, 20 to 40.361518, 30 fc rgb "#FF00FF"
set object 38 rect from 0.035042, 20 to 42.003946, 30 fc rgb "#808080"
set object 39 rect from 0.036506, 20 to 42.939014, 30 fc rgb "#800080"
set object 40 rect from 0.037712, 20 to 44.458925, 30 fc rgb "#008080"
set object 41 rect from 0.038620, 20 to 145.237895, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.130431, 30 to 155.169922, 40 fc rgb "#FF0000"
set object 43 rect from 0.064371, 30 to 76.705375, 40 fc rgb "#00FF00"
set object 44 rect from 0.066748, 30 to 77.531791, 40 fc rgb "#0000FF"
set object 45 rect from 0.067189, 30 to 78.508466, 40 fc rgb "#FFFF00"
set object 46 rect from 0.068070, 30 to 79.900086, 40 fc rgb "#FF00FF"
set object 47 rect from 0.069269, 30 to 81.319440, 40 fc rgb "#808080"
set object 48 rect from 0.070484, 30 to 82.036053, 40 fc rgb "#800080"
set object 49 rect from 0.071356, 30 to 82.877496, 40 fc rgb "#008080"
set object 50 rect from 0.071838, 30 to 150.028802, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.135118, 40 to 160.425472, 50 fc rgb "#FF0000"
set object 52 rect from 0.112683, 40 to 132.306489, 50 fc rgb "#00FF00"
set object 53 rect from 0.114835, 40 to 132.993048, 50 fc rgb "#0000FF"
set object 54 rect from 0.115176, 40 to 133.941985, 50 fc rgb "#FFFF00"
set object 55 rect from 0.116006, 40 to 135.253849, 50 fc rgb "#FF00FF"
set object 56 rect from 0.117145, 40 to 136.694013, 50 fc rgb "#808080"
set object 57 rect from 0.118393, 40 to 137.417558, 50 fc rgb "#800080"
set object 58 rect from 0.119263, 40 to 138.123769, 50 fc rgb "#008080"
set object 59 rect from 0.119619, 40 to 155.612605, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.131965, 50 to 156.992660, 60 fc rgb "#FF0000"
set object 61 rect from 0.080209, 50 to 94.833385, 60 fc rgb "#00FF00"
set object 62 rect from 0.082420, 50 to 95.655176, 60 fc rgb "#0000FF"
set object 63 rect from 0.082871, 50 to 96.722006, 60 fc rgb "#FFFF00"
set object 64 rect from 0.083805, 50 to 98.128649, 60 fc rgb "#FF00FF"
set object 65 rect from 0.085029, 50 to 99.477500, 60 fc rgb "#808080"
set object 66 rect from 0.086183, 50 to 100.157131, 60 fc rgb "#800080"
set object 67 rect from 0.087025, 50 to 100.837911, 60 fc rgb "#008080"
set object 68 rect from 0.087359, 50 to 151.924360, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0