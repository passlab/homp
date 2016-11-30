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

set object 15 rect from 0.172135, 0 to 155.233647, 10 fc rgb "#FF0000"
set object 16 rect from 0.131713, 0 to 118.103027, 10 fc rgb "#00FF00"
set object 17 rect from 0.134295, 0 to 118.893377, 10 fc rgb "#0000FF"
set object 18 rect from 0.134892, 0 to 119.540847, 10 fc rgb "#FFFF00"
set object 19 rect from 0.135688, 0 to 120.602025, 10 fc rgb "#FF00FF"
set object 20 rect from 0.136870, 0 to 121.723195, 10 fc rgb "#808080"
set object 21 rect from 0.138118, 0 to 122.257753, 10 fc rgb "#800080"
set object 22 rect from 0.138970, 0 to 122.733215, 10 fc rgb "#008080"
set object 23 rect from 0.139270, 0 to 151.397359, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.167704, 10 to 151.955735, 20 fc rgb "#FF0000"
set object 25 rect from 0.073266, 10 to 66.739536, 20 fc rgb "#00FF00"
set object 26 rect from 0.076039, 10 to 67.820115, 20 fc rgb "#0000FF"
set object 27 rect from 0.077030, 10 to 68.793962, 20 fc rgb "#FFFF00"
set object 28 rect from 0.078159, 10 to 69.912476, 20 fc rgb "#FF00FF"
set object 29 rect from 0.079375, 10 to 70.981594, 20 fc rgb "#808080"
set object 30 rect from 0.080592, 10 to 71.486183, 20 fc rgb "#800080"
set object 31 rect from 0.081573, 10 to 72.160993, 20 fc rgb "#008080"
set object 32 rect from 0.081945, 10 to 147.345843, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.165696, 20 to 153.020410, 30 fc rgb "#FF0000"
set object 34 rect from 0.044279, 20 to 42.755870, 30 fc rgb "#00FF00"
set object 35 rect from 0.048984, 20 to 44.233386, 30 fc rgb "#0000FF"
set object 36 rect from 0.050277, 20 to 46.375144, 30 fc rgb "#FFFF00"
set object 37 rect from 0.052736, 20 to 48.498394, 30 fc rgb "#FF00FF"
set object 38 rect from 0.055114, 20 to 49.720119, 30 fc rgb "#808080"
set object 39 rect from 0.056533, 20 to 50.494616, 30 fc rgb "#800080"
set object 40 rect from 0.057834, 20 to 51.598120, 30 fc rgb "#008080"
set object 41 rect from 0.058647, 20 to 145.284345, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.170711, 30 to 153.890205, 40 fc rgb "#FF0000"
set object 43 rect from 0.114309, 30 to 102.574349, 40 fc rgb "#00FF00"
set object 44 rect from 0.116657, 30 to 103.257099, 40 fc rgb "#0000FF"
set object 45 rect from 0.117165, 30 to 104.114512, 40 fc rgb "#FFFF00"
set object 46 rect from 0.118164, 30 to 105.195117, 40 fc rgb "#FF00FF"
set object 47 rect from 0.119363, 30 to 106.077216, 40 fc rgb "#808080"
set object 48 rect from 0.120378, 30 to 106.541215, 40 fc rgb "#800080"
set object 49 rect from 0.121182, 30 to 107.241604, 40 fc rgb "#008080"
set object 50 rect from 0.121740, 30 to 150.089196, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.173332, 40 to 156.199528, 50 fc rgb "#FF0000"
set object 52 rect from 0.150461, 40 to 134.402627, 50 fc rgb "#00FF00"
set object 53 rect from 0.152707, 40 to 135.099494, 50 fc rgb "#0000FF"
set object 54 rect from 0.153289, 40 to 135.946338, 50 fc rgb "#FFFF00"
set object 55 rect from 0.154240, 40 to 137.012801, 50 fc rgb "#FF00FF"
set object 56 rect from 0.155456, 40 to 137.926656, 50 fc rgb "#808080"
set object 57 rect from 0.156462, 40 to 138.354482, 50 fc rgb "#800080"
set object 58 rect from 0.157213, 40 to 138.969327, 50 fc rgb "#008080"
set object 59 rect from 0.157655, 40 to 152.558225, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.169298, 50 to 153.209243, 60 fc rgb "#FF0000"
set object 61 rect from 0.092497, 50 to 83.607239, 60 fc rgb "#00FF00"
set object 62 rect from 0.095146, 50 to 84.446118, 60 fc rgb "#0000FF"
set object 63 rect from 0.095836, 50 to 85.402325, 60 fc rgb "#FFFF00"
set object 64 rect from 0.096966, 50 to 86.614349, 60 fc rgb "#FF00FF"
set object 65 rect from 0.098349, 50 to 87.612907, 60 fc rgb "#808080"
set object 66 rect from 0.099462, 50 to 88.200386, 60 fc rgb "#800080"
set object 67 rect from 0.100372, 50 to 88.913131, 60 fc rgb "#008080"
set object 68 rect from 0.100937, 50 to 148.812790, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
