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

set object 15 rect from 0.100530, 0 to 145.935901, 10 fc rgb "#FF0000"
set object 16 rect from 0.090407, 0 to 130.431732, 10 fc rgb "#00FF00"
set object 17 rect from 0.092649, 0 to 131.289462, 10 fc rgb "#0000FF"
set object 18 rect from 0.093022, 0 to 132.477853, 10 fc rgb "#FFFF00"
set object 19 rect from 0.093867, 0 to 133.923421, 10 fc rgb "#FF00FF"
set object 20 rect from 0.094896, 0 to 134.518320, 10 fc rgb "#808080"
set object 21 rect from 0.095312, 0 to 135.254529, 10 fc rgb "#800080"
set object 22 rect from 0.096102, 0 to 135.996388, 10 fc rgb "#008080"
set object 23 rect from 0.096374, 0 to 140.648205, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.110148, 10 to 159.702001, 20 fc rgb "#FF0000"
set object 25 rect from 0.076379, 10 to 110.651621, 20 fc rgb "#00FF00"
set object 26 rect from 0.078651, 10 to 111.620984, 20 fc rgb "#0000FF"
set object 27 rect from 0.079103, 10 to 112.670893, 20 fc rgb "#FFFF00"
set object 28 rect from 0.079884, 10 to 114.281789, 20 fc rgb "#FF00FF"
set object 29 rect from 0.081007, 10 to 114.954410, 20 fc rgb "#808080"
set object 30 rect from 0.081473, 10 to 115.761271, 20 fc rgb "#800080"
set object 31 rect from 0.082310, 10 to 116.624654, 20 fc rgb "#008080"
set object 32 rect from 0.082650, 10 to 154.785941, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.108525, 20 to 157.216419, 30 fc rgb "#FF0000"
set object 34 rect from 0.063847, 20 to 92.529036, 30 fc rgb "#00FF00"
set object 35 rect from 0.065860, 20 to 93.391009, 30 fc rgb "#0000FF"
set object 36 rect from 0.066221, 20 to 94.511570, 30 fc rgb "#FFFF00"
set object 37 rect from 0.066996, 20 to 96.078661, 30 fc rgb "#FF00FF"
set object 38 rect from 0.068107, 20 to 96.666496, 30 fc rgb "#808080"
set object 39 rect from 0.068526, 20 to 97.397054, 30 fc rgb "#800080"
set object 40 rect from 0.069378, 20 to 98.383375, 30 fc rgb "#008080"
set object 41 rect from 0.069751, 20 to 152.677646, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.107001, 30 to 155.228232, 40 fc rgb "#FF0000"
set object 43 rect from 0.050040, 30 to 73.201107, 40 fc rgb "#00FF00"
set object 44 rect from 0.052176, 30 to 74.053186, 40 fc rgb "#0000FF"
set object 45 rect from 0.052516, 30 to 75.295274, 40 fc rgb "#FFFF00"
set object 46 rect from 0.053411, 30 to 76.849647, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054499, 30 to 77.481287, 40 fc rgb "#808080"
set object 48 rect from 0.054950, 30 to 78.259888, 40 fc rgb "#800080"
set object 49 rect from 0.055850, 30 to 79.254686, 40 fc rgb "#008080"
set object 50 rect from 0.056207, 30 to 150.504348, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.105424, 40 to 153.211782, 50 fc rgb "#FF0000"
set object 52 rect from 0.036260, 40 to 54.107746, 50 fc rgb "#00FF00"
set object 53 rect from 0.038659, 40 to 54.995152, 50 fc rgb "#0000FF"
set object 54 rect from 0.039031, 40 to 56.271153, 50 fc rgb "#FFFF00"
set object 55 rect from 0.039967, 40 to 57.925853, 50 fc rgb "#FF00FF"
set object 56 rect from 0.041148, 40 to 58.633801, 50 fc rgb "#808080"
set object 57 rect from 0.041704, 40 to 59.574903, 50 fc rgb "#800080"
set object 58 rect from 0.042532, 40 to 60.451005, 50 fc rgb "#008080"
set object 59 rect from 0.042916, 40 to 148.089417, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.103626, 50 to 154.076582, 60 fc rgb "#FF0000"
set object 61 rect from 0.017283, 50 to 28.412539, 60 fc rgb "#00FF00"
set object 62 rect from 0.020628, 50 to 30.481269, 60 fc rgb "#0000FF"
set object 63 rect from 0.021721, 50 to 33.339901, 60 fc rgb "#FFFF00"
set object 64 rect from 0.023776, 50 to 35.499068, 60 fc rgb "#FF00FF"
set object 65 rect from 0.025253, 50 to 36.448648, 60 fc rgb "#808080"
set object 66 rect from 0.025928, 50 to 37.464645, 60 fc rgb "#800080"
set object 67 rect from 0.027234, 50 to 39.093911, 60 fc rgb "#008080"
set object 68 rect from 0.027803, 50 to 145.357957, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
