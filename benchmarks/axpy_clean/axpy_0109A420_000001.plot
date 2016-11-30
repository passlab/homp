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

set object 15 rect from 0.156625, 0 to 157.228892, 10 fc rgb "#FF0000"
set object 16 rect from 0.097187, 0 to 94.991061, 10 fc rgb "#00FF00"
set object 17 rect from 0.098915, 0 to 95.516256, 10 fc rgb "#0000FF"
set object 18 rect from 0.099227, 0 to 96.080939, 10 fc rgb "#FFFF00"
set object 19 rect from 0.099828, 0 to 96.969427, 10 fc rgb "#FF00FF"
set object 20 rect from 0.100741, 0 to 101.447475, 10 fc rgb "#808080"
set object 21 rect from 0.105395, 0 to 101.958196, 10 fc rgb "#800080"
set object 22 rect from 0.106212, 0 to 102.508433, 10 fc rgb "#008080"
set object 23 rect from 0.106492, 0 to 150.371607, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.153398, 10 to 154.147166, 20 fc rgb "#FF0000"
set object 25 rect from 0.080307, 10 to 78.916539, 20 fc rgb "#00FF00"
set object 26 rect from 0.082220, 10 to 79.549659, 20 fc rgb "#0000FF"
set object 27 rect from 0.082661, 10 to 80.141338, 20 fc rgb "#FFFF00"
set object 28 rect from 0.083335, 10 to 81.233140, 20 fc rgb "#FF00FF"
set object 29 rect from 0.084420, 10 to 85.426916, 20 fc rgb "#808080"
set object 30 rect from 0.088781, 10 to 85.967518, 20 fc rgb "#800080"
set object 31 rect from 0.089568, 10 to 86.497523, 20 fc rgb "#008080"
set object 32 rect from 0.089871, 10 to 147.245553, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.155085, 20 to 158.805396, 30 fc rgb "#FF0000"
set object 34 rect from 0.059765, 20 to 59.243783, 30 fc rgb "#00FF00"
set object 35 rect from 0.061808, 20 to 59.933752, 30 fc rgb "#0000FF"
set object 36 rect from 0.062301, 20 to 62.627129, 30 fc rgb "#FFFF00"
set object 37 rect from 0.065130, 20 to 64.696073, 30 fc rgb "#FF00FF"
set object 38 rect from 0.067255, 20 to 68.719274, 30 fc rgb "#808080"
set object 39 rect from 0.071446, 20 to 69.296492, 30 fc rgb "#800080"
set object 40 rect from 0.072285, 20 to 69.884323, 30 fc rgb "#008080"
set object 41 rect from 0.072659, 20 to 148.815322, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.158212, 30 to 161.544077, 40 fc rgb "#FF0000"
set object 43 rect from 0.114107, 30 to 111.275661, 40 fc rgb "#00FF00"
set object 44 rect from 0.115797, 30 to 111.887572, 40 fc rgb "#0000FF"
set object 45 rect from 0.116216, 30 to 114.659007, 40 fc rgb "#FFFF00"
set object 46 rect from 0.119119, 30 to 116.479333, 40 fc rgb "#FF00FF"
set object 47 rect from 0.120983, 30 to 120.451471, 40 fc rgb "#808080"
set object 48 rect from 0.125145, 30 to 121.046036, 40 fc rgb "#800080"
set object 49 rect from 0.125994, 30 to 121.698412, 40 fc rgb "#008080"
set object 50 rect from 0.126399, 30 to 151.887414, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.159558, 40 to 162.015309, 50 fc rgb "#FF0000"
set object 52 rect from 0.133939, 40 to 130.226657, 50 fc rgb "#00FF00"
set object 53 rect from 0.135464, 40 to 130.746065, 50 fc rgb "#0000FF"
set object 54 rect from 0.135785, 40 to 133.219729, 50 fc rgb "#FFFF00"
set object 55 rect from 0.138359, 40 to 134.582320, 50 fc rgb "#FF00FF"
set object 56 rect from 0.139768, 40 to 138.592999, 50 fc rgb "#808080"
set object 57 rect from 0.143933, 40 to 139.117218, 50 fc rgb "#800080"
set object 58 rect from 0.144728, 40 to 139.653010, 50 fc rgb "#008080"
set object 59 rect from 0.145032, 40 to 153.339622, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.151509, 50 to 158.075924, 60 fc rgb "#FF0000"
set object 61 rect from 0.033671, 50 to 35.310753, 60 fc rgb "#00FF00"
set object 62 rect from 0.037066, 50 to 36.398706, 60 fc rgb "#0000FF"
set object 63 rect from 0.037894, 50 to 40.222441, 60 fc rgb "#FFFF00"
set object 64 rect from 0.041887, 50 to 43.100838, 60 fc rgb "#FF00FF"
set object 65 rect from 0.044847, 50 to 47.501791, 60 fc rgb "#808080"
set object 66 rect from 0.049431, 50 to 48.185011, 60 fc rgb "#800080"
set object 67 rect from 0.050516, 50 to 49.285501, 60 fc rgb "#008080"
set object 68 rect from 0.051266, 50 to 145.098551, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
