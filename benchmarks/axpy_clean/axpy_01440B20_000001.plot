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

set object 15 rect from 0.209104, 0 to 156.808136, 10 fc rgb "#FF0000"
set object 16 rect from 0.102400, 0 to 73.773670, 10 fc rgb "#00FF00"
set object 17 rect from 0.104382, 0 to 74.399752, 10 fc rgb "#0000FF"
set object 18 rect from 0.105040, 0 to 74.896079, 10 fc rgb "#FFFF00"
set object 19 rect from 0.105806, 0 to 75.680994, 10 fc rgb "#FF00FF"
set object 20 rect from 0.106875, 0 to 82.455178, 10 fc rgb "#808080"
set object 21 rect from 0.116426, 0 to 82.875634, 10 fc rgb "#800080"
set object 22 rect from 0.117278, 0 to 83.287595, 10 fc rgb "#008080"
set object 23 rect from 0.117574, 0 to 147.747282, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.210932, 10 to 157.529932, 20 fc rgb "#FF0000"
set object 25 rect from 0.125623, 10 to 90.056106, 20 fc rgb "#00FF00"
set object 26 rect from 0.127341, 10 to 90.467348, 20 fc rgb "#0000FF"
set object 27 rect from 0.127697, 10 to 90.889939, 20 fc rgb "#FFFF00"
set object 28 rect from 0.128323, 10 to 91.563524, 20 fc rgb "#FF00FF"
set object 29 rect from 0.129246, 10 to 98.166831, 20 fc rgb "#808080"
set object 30 rect from 0.138573, 10 to 98.520640, 20 fc rgb "#800080"
set object 31 rect from 0.139315, 10 to 98.870201, 20 fc rgb "#008080"
set object 32 rect from 0.139570, 10 to 148.947697, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.207301, 20 to 160.393057, 30 fc rgb "#FF0000"
set object 34 rect from 0.072034, 20 to 52.252118, 30 fc rgb "#00FF00"
set object 35 rect from 0.074021, 20 to 52.870403, 30 fc rgb "#0000FF"
set object 36 rect from 0.074673, 20 to 54.938683, 30 fc rgb "#FFFF00"
set object 37 rect from 0.077616, 20 to 58.948318, 30 fc rgb "#FF00FF"
set object 38 rect from 0.083279, 20 to 65.633160, 30 fc rgb "#808080"
set object 39 rect from 0.092688, 20 to 66.160687, 30 fc rgb "#800080"
set object 40 rect from 0.093675, 20 to 66.624398, 30 fc rgb "#008080"
set object 41 rect from 0.094087, 20 to 146.491578, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.205504, 30 to 161.362309, 40 fc rgb "#FF0000"
set object 43 rect from 0.035415, 30 to 27.233582, 40 fc rgb "#00FF00"
set object 44 rect from 0.038830, 30 to 28.130523, 40 fc rgb "#0000FF"
set object 45 rect from 0.039794, 30 to 31.360917, 40 fc rgb "#FFFF00"
set object 46 rect from 0.044377, 30 to 36.038464, 40 fc rgb "#FF00FF"
set object 47 rect from 0.050949, 30 to 42.870082, 40 fc rgb "#808080"
set object 48 rect from 0.060593, 30 to 43.464965, 40 fc rgb "#800080"
set object 49 rect from 0.061812, 30 to 44.315819, 40 fc rgb "#008080"
set object 50 rect from 0.062626, 30 to 144.770017, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.214067, 40 to 164.764298, 50 fc rgb "#FF0000"
set object 52 rect from 0.177549, 40 to 126.827723, 50 fc rgb "#00FF00"
set object 53 rect from 0.179200, 40 to 127.368003, 50 fc rgb "#0000FF"
set object 54 rect from 0.179742, 40 to 129.610001, 50 fc rgb "#FFFF00"
set object 55 rect from 0.182919, 40 to 133.234617, 50 fc rgb "#FF00FF"
set object 56 rect from 0.188016, 40 to 139.803186, 50 fc rgb "#808080"
set object 57 rect from 0.197292, 40 to 140.291007, 50 fc rgb "#800080"
set object 58 rect from 0.198229, 40 to 140.730608, 50 fc rgb "#008080"
set object 59 rect from 0.198589, 40 to 151.470464, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.212675, 50 to 164.079355, 60 fc rgb "#FF0000"
set object 61 rect from 0.147330, 50 to 105.410401, 60 fc rgb "#00FF00"
set object 62 rect from 0.148993, 50 to 105.886167, 60 fc rgb "#0000FF"
set object 63 rect from 0.149445, 50 to 107.979973, 60 fc rgb "#FFFF00"
set object 64 rect from 0.152446, 50 to 112.017966, 60 fc rgb "#FF00FF"
set object 65 rect from 0.158107, 50 to 118.588658, 60 fc rgb "#808080"
set object 66 rect from 0.167387, 50 to 119.219696, 60 fc rgb "#800080"
set object 67 rect from 0.168484, 50 to 119.694046, 60 fc rgb "#008080"
set object 68 rect from 0.168932, 50 to 150.224681, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
