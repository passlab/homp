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

set object 15 rect from 0.117638, 0 to 162.226050, 10 fc rgb "#FF0000"
set object 16 rect from 0.096959, 0 to 132.368091, 10 fc rgb "#00FF00"
set object 17 rect from 0.099104, 0 to 133.192527, 10 fc rgb "#0000FF"
set object 18 rect from 0.099468, 0 to 134.184529, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100229, 0 to 135.570653, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101244, 0 to 137.195394, 10 fc rgb "#808080"
set object 21 rect from 0.102457, 0 to 137.861644, 10 fc rgb "#800080"
set object 22 rect from 0.103205, 0 to 138.564090, 10 fc rgb "#008080"
set object 23 rect from 0.103478, 0 to 156.942961, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.116074, 10 to 160.704530, 20 fc rgb "#FF0000"
set object 25 rect from 0.082376, 10 to 113.065584, 20 fc rgb "#00FF00"
set object 26 rect from 0.084715, 10 to 114.005306, 20 fc rgb "#0000FF"
set object 27 rect from 0.085172, 10 to 115.077742, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086007, 10 to 116.744039, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087232, 10 to 118.485407, 20 fc rgb "#808080"
set object 30 rect from 0.088520, 10 to 119.280350, 20 fc rgb "#800080"
set object 31 rect from 0.089350, 10 to 120.057867, 20 fc rgb "#008080"
set object 32 rect from 0.089677, 10 to 154.705592, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.114339, 20 to 157.914856, 30 fc rgb "#FF0000"
set object 34 rect from 0.069043, 20 to 94.877070, 30 fc rgb "#00FF00"
set object 35 rect from 0.071121, 20 to 95.622413, 30 fc rgb "#0000FF"
set object 36 rect from 0.071445, 20 to 96.745790, 30 fc rgb "#FFFF00"
set object 37 rect from 0.072294, 20 to 98.215027, 30 fc rgb "#FF00FF"
set object 38 rect from 0.073378, 20 to 99.757994, 30 fc rgb "#808080"
set object 39 rect from 0.074531, 20 to 100.451056, 30 fc rgb "#800080"
set object 40 rect from 0.075300, 20 to 101.260745, 30 fc rgb "#008080"
set object 41 rect from 0.075653, 20 to 152.579488, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.112759, 30 to 155.878569, 40 fc rgb "#FF0000"
set object 43 rect from 0.054387, 30 to 75.384205, 40 fc rgb "#00FF00"
set object 44 rect from 0.056604, 30 to 76.211322, 40 fc rgb "#0000FF"
set object 45 rect from 0.056975, 30 to 77.395023, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057847, 30 to 78.870963, 40 fc rgb "#FF00FF"
set object 47 rect from 0.058948, 30 to 80.391141, 40 fc rgb "#808080"
set object 48 rect from 0.060094, 30 to 81.088224, 40 fc rgb "#800080"
set object 49 rect from 0.060878, 30 to 81.948855, 40 fc rgb "#008080"
set object 50 rect from 0.061265, 30 to 150.352843, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.111129, 40 to 154.078217, 50 fc rgb "#FF0000"
set object 52 rect from 0.039942, 40 to 56.364553, 50 fc rgb "#00FF00"
set object 53 rect from 0.042433, 40 to 57.226524, 50 fc rgb "#0000FF"
set object 54 rect from 0.042818, 40 to 58.466528, 50 fc rgb "#FFFF00"
set object 55 rect from 0.043732, 40 to 60.053733, 50 fc rgb "#FF00FF"
set object 56 rect from 0.044924, 40 to 61.648982, 50 fc rgb "#808080"
set object 57 rect from 0.046108, 40 to 62.489504, 50 fc rgb "#800080"
set object 58 rect from 0.046991, 40 to 63.496253, 50 fc rgb "#008080"
set object 59 rect from 0.047497, 40 to 148.199927, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.109417, 50 to 155.448254, 60 fc rgb "#FF0000"
set object 61 rect from 0.019856, 50 to 30.325808, 60 fc rgb "#00FF00"
set object 62 rect from 0.023145, 50 to 32.213296, 60 fc rgb "#0000FF"
set object 63 rect from 0.024160, 50 to 35.024418, 60 fc rgb "#FFFF00"
set object 64 rect from 0.026289, 50 to 37.260447, 60 fc rgb "#FF00FF"
set object 65 rect from 0.027930, 50 to 39.265902, 60 fc rgb "#808080"
set object 66 rect from 0.029436, 50 to 40.406706, 60 fc rgb "#800080"
set object 67 rect from 0.030674, 50 to 42.005977, 60 fc rgb "#008080"
set object 68 rect from 0.031485, 50 to 145.320437, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
