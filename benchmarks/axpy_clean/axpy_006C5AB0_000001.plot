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

set object 15 rect from 0.506439, 0 to 160.314718, 10 fc rgb "#FF0000"
set object 16 rect from 0.441359, 0 to 129.196565, 10 fc rgb "#00FF00"
set object 17 rect from 0.443614, 0 to 129.395063, 10 fc rgb "#0000FF"
set object 18 rect from 0.444042, 0 to 129.618622, 10 fc rgb "#FFFF00"
set object 19 rect from 0.444826, 0 to 129.927293, 10 fc rgb "#FF00FF"
set object 20 rect from 0.445884, 0 to 141.941017, 10 fc rgb "#808080"
set object 21 rect from 0.487115, 0 to 142.116782, 10 fc rgb "#800080"
set object 22 rect from 0.487956, 0 to 142.278840, 10 fc rgb "#008080"
set object 23 rect from 0.488272, 0 to 147.442025, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.500937, 10 to 158.832587, 20 fc rgb "#FF0000"
set object 25 rect from 0.215984, 10 to 63.661043, 20 fc rgb "#00FF00"
set object 26 rect from 0.218797, 10 to 63.871494, 20 fc rgb "#0000FF"
set object 27 rect from 0.219243, 10 to 64.135567, 20 fc rgb "#FFFF00"
set object 28 rect from 0.220204, 10 to 64.487671, 20 fc rgb "#FF00FF"
set object 29 rect from 0.221399, 10 to 76.538408, 20 fc rgb "#808080"
set object 30 rect from 0.262756, 10 to 76.736029, 20 fc rgb "#800080"
set object 31 rect from 0.263659, 10 to 76.914409, 20 fc rgb "#008080"
set object 32 rect from 0.264000, 10 to 145.699296, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.504807, 20 to 168.013461, 30 fc rgb "#FF0000"
set object 34 rect from 0.356550, 20 to 104.488816, 30 fc rgb "#00FF00"
set object 35 rect from 0.358829, 20 to 104.661958, 30 fc rgb "#0000FF"
set object 36 rect from 0.359210, 20 to 105.574271, 30 fc rgb "#FFFF00"
set object 37 rect from 0.362327, 20 to 113.328375, 30 fc rgb "#FF00FF"
set object 38 rect from 0.388943, 20 to 125.162554, 30 fc rgb "#808080"
set object 39 rect from 0.429551, 20 to 125.585194, 30 fc rgb "#800080"
set object 40 rect from 0.431246, 20 to 125.796218, 30 fc rgb "#008080"
set object 41 rect from 0.431721, 20 to 146.999276, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.503087, 30 to 166.703856, 40 fc rgb "#FF0000"
set object 43 rect from 0.275524, 30 to 80.857488, 40 fc rgb "#00FF00"
set object 44 rect from 0.277751, 30 to 81.063266, 40 fc rgb "#0000FF"
set object 45 rect from 0.278228, 30 to 82.002108, 40 fc rgb "#FFFF00"
set object 46 rect from 0.281461, 30 to 88.958154, 40 fc rgb "#FF00FF"
set object 47 rect from 0.305337, 30 to 100.738989, 40 fc rgb "#808080"
set object 48 rect from 0.345758, 30 to 101.141232, 40 fc rgb "#800080"
set object 49 rect from 0.347386, 30 to 101.353715, 40 fc rgb "#008080"
set object 50 rect from 0.347858, 30 to 146.435853, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.496480, 40 to 165.568844, 50 fc rgb "#FF0000"
set object 52 rect from 0.041268, 40 to 13.086062, 50 fc rgb "#00FF00"
set object 53 rect from 0.045351, 40 to 13.467596, 50 fc rgb "#0000FF"
set object 54 rect from 0.046336, 40 to 14.801680, 50 fc rgb "#FFFF00"
set object 55 rect from 0.050973, 40 to 21.839050, 50 fc rgb "#FF00FF"
set object 56 rect from 0.075057, 40 to 33.711703, 50 fc rgb "#808080"
set object 57 rect from 0.115794, 40 to 34.211288, 50 fc rgb "#800080"
set object 58 rect from 0.117993, 40 to 34.622852, 50 fc rgb "#008080"
set object 59 rect from 0.118912, 40 to 144.396692, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.498579, 50 to 166.058518, 60 fc rgb "#FF0000"
set object 61 rect from 0.130713, 50 to 38.792982, 60 fc rgb "#00FF00"
set object 62 rect from 0.133443, 50 to 38.990316, 60 fc rgb "#0000FF"
set object 63 rect from 0.133883, 50 to 39.865025, 60 fc rgb "#FFFF00"
set object 64 rect from 0.136895, 50 to 47.672473, 60 fc rgb "#FF00FF"
set object 65 rect from 0.163707, 50 to 59.350424, 60 fc rgb "#808080"
set object 66 rect from 0.203751, 50 to 59.749740, 60 fc rgb "#800080"
set object 67 rect from 0.205411, 50 to 60.012945, 60 fc rgb "#008080"
set object 68 rect from 0.206044, 50 to 145.121592, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
