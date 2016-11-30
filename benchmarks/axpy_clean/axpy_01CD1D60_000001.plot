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

set object 15 rect from 0.214296, 0 to 156.460946, 10 fc rgb "#FF0000"
set object 16 rect from 0.106521, 0 to 75.009358, 10 fc rgb "#00FF00"
set object 17 rect from 0.108757, 0 to 75.497252, 10 fc rgb "#0000FF"
set object 18 rect from 0.109214, 0 to 76.003131, 10 fc rgb "#FFFF00"
set object 19 rect from 0.109976, 0 to 76.845355, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111200, 0 to 83.275138, 10 fc rgb "#808080"
set object 21 rect from 0.120468, 0 to 83.716658, 10 fc rgb "#800080"
set object 22 rect from 0.121373, 0 to 84.146422, 10 fc rgb "#008080"
set object 23 rect from 0.121733, 0 to 147.702435, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.253214, 10 to 183.002268, 20 fc rgb "#FF0000"
set object 25 rect from 0.159463, 10 to 111.365240, 20 fc rgb "#00FF00"
set object 26 rect from 0.161284, 10 to 111.848287, 20 fc rgb "#0000FF"
set object 27 rect from 0.161742, 10 to 112.336181, 20 fc rgb "#FFFF00"
set object 28 rect from 0.162445, 10 to 113.080129, 20 fc rgb "#FF00FF"
set object 29 rect from 0.163532, 10 to 119.279464, 20 fc rgb "#808080"
set object 30 rect from 0.172491, 10 to 119.650397, 20 fc rgb "#800080"
set object 31 rect from 0.173278, 10 to 120.040015, 20 fc rgb "#008080"
set object 32 rect from 0.173586, 10 to 174.583536, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.216269, 20 to 162.525346, 30 fc rgb "#FF0000"
set object 34 rect from 0.130555, 20 to 91.437116, 30 fc rgb "#00FF00"
set object 35 rect from 0.132498, 20 to 91.912553, 30 fc rgb "#0000FF"
set object 36 rect from 0.132934, 20 to 94.127092, 30 fc rgb "#FFFF00"
set object 37 rect from 0.136135, 20 to 98.139566, 30 fc rgb "#FF00FF"
set object 38 rect from 0.141961, 20 to 104.325062, 30 fc rgb "#808080"
set object 39 rect from 0.150906, 20 to 104.844100, 30 fc rgb "#800080"
set object 40 rect from 0.151906, 20 to 105.320918, 30 fc rgb "#008080"
set object 41 rect from 0.152314, 20 to 149.187567, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.256485, 30 to 190.155942, 40 fc rgb "#FF0000"
set object 43 rect from 0.182162, 30 to 127.021371, 40 fc rgb "#00FF00"
set object 44 rect from 0.183910, 30 to 127.497499, 40 fc rgb "#0000FF"
set object 45 rect from 0.184352, 30 to 129.640069, 40 fc rgb "#FFFF00"
set object 46 rect from 0.187450, 30 to 133.532818, 40 fc rgb "#FF00FF"
set object 47 rect from 0.193076, 30 to 139.712776, 40 fc rgb "#808080"
set object 48 rect from 0.202022, 30 to 140.193060, 40 fc rgb "#800080"
set object 49 rect from 0.202995, 30 to 140.674715, 40 fc rgb "#008080"
set object 50 rect from 0.203420, 30 to 177.079044, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.210091, 40 to 160.835381, 50 fc rgb "#FF0000"
set object 52 rect from 0.036585, 40 to 27.492175, 50 fc rgb "#00FF00"
set object 53 rect from 0.040249, 40 to 28.629897, 50 fc rgb "#0000FF"
set object 54 rect from 0.041502, 40 to 31.679741, 50 fc rgb "#FFFF00"
set object 55 rect from 0.045938, 40 to 36.562117, 50 fc rgb "#FF00FF"
set object 56 rect from 0.052964, 40 to 43.016815, 50 fc rgb "#808080"
set object 57 rect from 0.062307, 40 to 43.602975, 50 fc rgb "#800080"
set object 58 rect from 0.063579, 40 to 44.491562, 50 fc rgb "#008080"
set object 59 rect from 0.064422, 40 to 144.807616, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.212088, 50 to 159.615996, 60 fc rgb "#FF0000"
set object 61 rect from 0.075170, 50 to 53.344204, 60 fc rgb "#00FF00"
set object 62 rect from 0.077483, 50 to 53.836244, 60 fc rgb "#0000FF"
set object 63 rect from 0.077912, 50 to 56.070160, 60 fc rgb "#FFFF00"
set object 64 rect from 0.081186, 50 to 60.000981, 60 fc rgb "#FF00FF"
set object 65 rect from 0.086844, 50 to 66.239070, 60 fc rgb "#808080"
set object 66 rect from 0.095868, 50 to 66.769183, 60 fc rgb "#800080"
set object 67 rect from 0.096911, 50 to 67.273680, 60 fc rgb "#008080"
set object 68 rect from 0.097354, 50 to 146.311423, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
