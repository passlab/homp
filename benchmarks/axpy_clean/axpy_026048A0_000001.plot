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

set object 15 rect from 0.200611, 0 to 156.233664, 10 fc rgb "#FF0000"
set object 16 rect from 0.130743, 0 to 100.767361, 10 fc rgb "#00FF00"
set object 17 rect from 0.133125, 0 to 101.407243, 10 fc rgb "#0000FF"
set object 18 rect from 0.133709, 0 to 102.137453, 10 fc rgb "#FFFF00"
set object 19 rect from 0.134696, 0 to 102.966339, 10 fc rgb "#FF00FF"
set object 20 rect from 0.135789, 0 to 104.858658, 10 fc rgb "#808080"
set object 21 rect from 0.138279, 0 to 105.320162, 10 fc rgb "#800080"
set object 22 rect from 0.139132, 0 to 105.737641, 10 fc rgb "#008080"
set object 23 rect from 0.139420, 0 to 151.579147, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.195605, 10 to 152.863467, 20 fc rgb "#FF0000"
set object 25 rect from 0.084170, 10 to 65.563988, 20 fc rgb "#00FF00"
set object 26 rect from 0.086739, 10 to 66.367066, 20 fc rgb "#0000FF"
set object 27 rect from 0.087564, 10 to 67.044141, 20 fc rgb "#FFFF00"
set object 28 rect from 0.088466, 10 to 68.009657, 20 fc rgb "#FF00FF"
set object 29 rect from 0.089743, 10 to 70.078077, 20 fc rgb "#808080"
set object 30 rect from 0.092468, 10 to 70.567667, 20 fc rgb "#800080"
set object 31 rect from 0.093411, 10 to 71.076232, 20 fc rgb "#008080"
set object 32 rect from 0.093797, 10 to 147.893184, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.193082, 20 to 156.670117, 30 fc rgb "#FF0000"
set object 34 rect from 0.051105, 20 to 41.701629, 30 fc rgb "#00FF00"
set object 35 rect from 0.055534, 20 to 43.580285, 30 fc rgb "#0000FF"
set object 36 rect from 0.057548, 20 to 48.031374, 30 fc rgb "#FFFF00"
set object 37 rect from 0.063439, 20 to 50.262232, 30 fc rgb "#FF00FF"
set object 38 rect from 0.066348, 20 to 51.999704, 30 fc rgb "#808080"
set object 39 rect from 0.068798, 20 to 52.707902, 30 fc rgb "#800080"
set object 40 rect from 0.070005, 20 to 53.608898, 30 fc rgb "#008080"
set object 41 rect from 0.070759, 20 to 145.379960, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.198039, 30 to 156.064393, 40 fc rgb "#FF0000"
set object 43 rect from 0.105989, 30 to 81.891224, 40 fc rgb "#00FF00"
set object 44 rect from 0.108255, 30 to 82.614602, 40 fc rgb "#0000FF"
set object 45 rect from 0.108949, 30 to 84.736155, 40 fc rgb "#FFFF00"
set object 46 rect from 0.111808, 30 to 86.097896, 40 fc rgb "#FF00FF"
set object 47 rect from 0.113553, 30 to 87.691908, 40 fc rgb "#808080"
set object 48 rect from 0.115663, 30 to 88.242223, 40 fc rgb "#800080"
set object 49 rect from 0.116723, 30 to 88.838079, 40 fc rgb "#008080"
set object 50 rect from 0.117155, 30 to 149.556270, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.204167, 40 to 160.091932, 50 fc rgb "#FF0000"
set object 52 rect from 0.173231, 40 to 132.792578, 50 fc rgb "#00FF00"
set object 53 rect from 0.175301, 40 to 133.430943, 50 fc rgb "#0000FF"
set object 54 rect from 0.175897, 40 to 135.166139, 50 fc rgb "#FFFF00"
set object 55 rect from 0.178185, 40 to 136.472468, 50 fc rgb "#FF00FF"
set object 56 rect from 0.179906, 40 to 137.988299, 50 fc rgb "#808080"
set object 57 rect from 0.181911, 40 to 138.446008, 50 fc rgb "#800080"
set object 58 rect from 0.182793, 40 to 139.056286, 50 fc rgb "#008080"
set object 59 rect from 0.183331, 40 to 154.616879, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.202691, 50 to 159.596269, 60 fc rgb "#FF0000"
set object 61 rect from 0.149987, 50 to 115.189374, 60 fc rgb "#00FF00"
set object 62 rect from 0.152139, 50 to 115.942354, 60 fc rgb "#0000FF"
set object 63 rect from 0.152860, 50 to 118.041137, 60 fc rgb "#FFFF00"
set object 64 rect from 0.155697, 50 to 119.490928, 60 fc rgb "#FF00FF"
set object 65 rect from 0.157550, 50 to 121.038638, 60 fc rgb "#808080"
set object 66 rect from 0.159585, 50 to 121.547204, 60 fc rgb "#800080"
set object 67 rect from 0.160543, 50 to 122.052734, 60 fc rgb "#008080"
set object 68 rect from 0.160913, 50 to 153.196690, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
