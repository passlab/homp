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

set object 15 rect from 0.098100, 0 to 161.751659, 10 fc rgb "#FF0000"
set object 16 rect from 0.081525, 0 to 132.926921, 10 fc rgb "#00FF00"
set object 17 rect from 0.083061, 0 to 133.734842, 10 fc rgb "#0000FF"
set object 18 rect from 0.083362, 0 to 134.613416, 10 fc rgb "#FFFF00"
set object 19 rect from 0.083910, 0 to 135.980294, 10 fc rgb "#FF00FF"
set object 20 rect from 0.084761, 0 to 137.372830, 10 fc rgb "#808080"
set object 21 rect from 0.085643, 0 to 138.106890, 10 fc rgb "#800080"
set object 22 rect from 0.086301, 0 to 138.821611, 10 fc rgb "#008080"
set object 23 rect from 0.086531, 0 to 156.810984, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.096753, 10 to 160.126149, 20 fc rgb "#FF0000"
set object 25 rect from 0.069430, 10 to 113.631702, 20 fc rgb "#00FF00"
set object 26 rect from 0.071065, 10 to 114.529567, 20 fc rgb "#0000FF"
set object 27 rect from 0.071407, 10 to 115.535040, 20 fc rgb "#FFFF00"
set object 28 rect from 0.072067, 10 to 117.170172, 20 fc rgb "#FF00FF"
set object 29 rect from 0.073060, 10 to 118.628574, 20 fc rgb "#808080"
set object 30 rect from 0.073983, 10 to 119.470242, 20 fc rgb "#800080"
set object 31 rect from 0.074732, 10 to 120.339195, 20 fc rgb "#008080"
set object 32 rect from 0.075028, 10 to 154.555910, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.095332, 20 to 157.432601, 30 fc rgb "#FF0000"
set object 34 rect from 0.058341, 20 to 95.499394, 30 fc rgb "#00FF00"
set object 35 rect from 0.059780, 20 to 96.284816, 30 fc rgb "#0000FF"
set object 36 rect from 0.060063, 20 to 97.349742, 30 fc rgb "#FFFF00"
set object 37 rect from 0.060709, 20 to 98.883632, 30 fc rgb "#FF00FF"
set object 38 rect from 0.061664, 20 to 100.128447, 30 fc rgb "#808080"
set object 39 rect from 0.062449, 20 to 100.857672, 30 fc rgb "#800080"
set object 40 rect from 0.063111, 20 to 101.755537, 30 fc rgb "#008080"
set object 41 rect from 0.063471, 20 to 152.527253, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.094121, 30 to 155.609586, 40 fc rgb "#FF0000"
set object 43 rect from 0.046085, 30 to 76.032375, 40 fc rgb "#00FF00"
set object 44 rect from 0.047645, 30 to 76.788885, 40 fc rgb "#0000FF"
set object 45 rect from 0.047911, 30 to 77.887510, 40 fc rgb "#FFFF00"
set object 46 rect from 0.048616, 30 to 79.549927, 40 fc rgb "#FF00FF"
set object 47 rect from 0.049625, 30 to 80.728874, 40 fc rgb "#808080"
set object 48 rect from 0.050371, 30 to 81.499840, 40 fc rgb "#800080"
set object 49 rect from 0.051073, 30 to 82.476401, 40 fc rgb "#008080"
set object 50 rect from 0.051467, 30 to 150.399078, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.092785, 40 to 153.770392, 50 fc rgb "#FF0000"
set object 52 rect from 0.033467, 40 to 56.146078, 50 fc rgb "#00FF00"
set object 53 rect from 0.035278, 40 to 57.003782, 50 fc rgb "#0000FF"
set object 54 rect from 0.035615, 40 to 58.071914, 50 fc rgb "#FFFF00"
set object 55 rect from 0.036270, 40 to 59.859602, 50 fc rgb "#FF00FF"
set object 56 rect from 0.037381, 40 to 61.167028, 50 fc rgb "#808080"
set object 57 rect from 0.038203, 40 to 61.981363, 50 fc rgb "#800080"
set object 58 rect from 0.038944, 40 to 62.933846, 50 fc rgb "#008080"
set object 59 rect from 0.039295, 40 to 148.171289, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.091326, 50 to 154.457972, 60 fc rgb "#FF0000"
set object 61 rect from 0.016133, 50 to 29.423858, 60 fc rgb "#00FF00"
set object 62 rect from 0.018709, 50 to 30.938505, 60 fc rgb "#0000FF"
set object 63 rect from 0.019385, 50 to 33.439335, 60 fc rgb "#FFFF00"
set object 64 rect from 0.020969, 50 to 36.010866, 60 fc rgb "#FF00FF"
set object 65 rect from 0.022536, 50 to 37.491767, 60 fc rgb "#808080"
set object 66 rect from 0.023470, 50 to 38.482783, 60 fc rgb "#800080"
set object 67 rect from 0.024416, 50 to 40.211067, 60 fc rgb "#008080"
set object 68 rect from 0.025152, 50 to 145.313840, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
