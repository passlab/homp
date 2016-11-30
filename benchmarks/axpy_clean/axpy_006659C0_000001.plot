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

set object 15 rect from 0.160112, 0 to 158.705018, 10 fc rgb "#FF0000"
set object 16 rect from 0.136482, 0 to 134.081603, 10 fc rgb "#00FF00"
set object 17 rect from 0.138638, 0 to 134.765129, 10 fc rgb "#0000FF"
set object 18 rect from 0.139136, 0 to 135.519431, 10 fc rgb "#FFFF00"
set object 19 rect from 0.139892, 0 to 136.508362, 10 fc rgb "#FF00FF"
set object 20 rect from 0.140909, 0 to 137.734831, 10 fc rgb "#808080"
set object 21 rect from 0.142177, 0 to 138.226388, 10 fc rgb "#800080"
set object 22 rect from 0.142949, 0 to 138.730548, 10 fc rgb "#008080"
set object 23 rect from 0.143204, 0 to 154.720207, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.158577, 10 to 157.522179, 20 fc rgb "#FF0000"
set object 25 rect from 0.120012, 10 to 118.256766, 20 fc rgb "#00FF00"
set object 26 rect from 0.122327, 10 to 119.041124, 20 fc rgb "#0000FF"
set object 27 rect from 0.122895, 10 to 119.838087, 20 fc rgb "#FFFF00"
set object 28 rect from 0.123746, 10 to 120.946271, 20 fc rgb "#FF00FF"
set object 29 rect from 0.124891, 10 to 122.209582, 20 fc rgb "#808080"
set object 30 rect from 0.126183, 10 to 122.788398, 20 fc rgb "#800080"
set object 31 rect from 0.127030, 10 to 123.370122, 20 fc rgb "#008080"
set object 32 rect from 0.127366, 10 to 153.118526, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.156859, 20 to 168.033934, 30 fc rgb "#FF0000"
set object 34 rect from 0.090119, 20 to 89.108508, 30 fc rgb "#00FF00"
set object 35 rect from 0.092278, 20 to 89.754222, 30 fc rgb "#0000FF"
set object 36 rect from 0.092687, 20 to 102.476529, 30 fc rgb "#FFFF00"
set object 37 rect from 0.105922, 20 to 104.102448, 30 fc rgb "#FF00FF"
set object 38 rect from 0.107489, 20 to 105.287226, 30 fc rgb "#808080"
set object 39 rect from 0.108709, 20 to 105.863133, 30 fc rgb "#800080"
set object 40 rect from 0.109600, 20 to 106.564111, 30 fc rgb "#008080"
set object 41 rect from 0.110029, 20 to 151.543993, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.155084, 30 to 154.195686, 40 fc rgb "#FF0000"
set object 43 rect from 0.074255, 30 to 73.759715, 40 fc rgb "#00FF00"
set object 44 rect from 0.076441, 30 to 74.487840, 40 fc rgb "#0000FF"
set object 45 rect from 0.076962, 30 to 75.361396, 40 fc rgb "#FFFF00"
set object 46 rect from 0.077860, 30 to 76.594651, 40 fc rgb "#FF00FF"
set object 47 rect from 0.079125, 30 to 77.681506, 40 fc rgb "#808080"
set object 48 rect from 0.080246, 30 to 78.336915, 40 fc rgb "#800080"
set object 49 rect from 0.081171, 30 to 78.918640, 40 fc rgb "#008080"
set object 50 rect from 0.081515, 30 to 149.729983, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.153263, 40 to 152.538741, 50 fc rgb "#FF0000"
set object 52 rect from 0.057986, 40 to 58.371172, 50 fc rgb "#00FF00"
set object 53 rect from 0.060579, 40 to 59.103175, 50 fc rgb "#0000FF"
set object 54 rect from 0.061076, 40 to 59.935041, 50 fc rgb "#FFFF00"
set object 55 rect from 0.061939, 40 to 61.263311, 50 fc rgb "#FF00FF"
set object 56 rect from 0.063316, 40 to 62.423850, 50 fc rgb "#808080"
set object 57 rect from 0.064525, 40 to 63.058899, 50 fc rgb "#800080"
set object 58 rect from 0.065449, 40 to 63.740486, 50 fc rgb "#008080"
set object 59 rect from 0.065880, 40 to 147.936333, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.151075, 50 to 154.144300, 60 fc rgb "#FF0000"
set object 61 rect from 0.033550, 50 to 35.622849, 60 fc rgb "#00FF00"
set object 62 rect from 0.037188, 50 to 36.792114, 60 fc rgb "#0000FF"
set object 63 rect from 0.038092, 50 to 39.938272, 60 fc rgb "#FFFF00"
set object 64 rect from 0.041362, 50 to 41.876383, 60 fc rgb "#FF00FF"
set object 65 rect from 0.043333, 50 to 43.355902, 60 fc rgb "#808080"
set object 66 rect from 0.044867, 50 to 44.194554, 60 fc rgb "#800080"
set object 67 rect from 0.046275, 50 to 45.623657, 60 fc rgb "#008080"
set object 68 rect from 0.047210, 50 to 145.396138, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
