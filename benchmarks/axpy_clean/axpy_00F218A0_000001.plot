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

set object 15 rect from 0.161699, 0 to 161.630839, 10 fc rgb "#FF0000"
set object 16 rect from 0.134169, 0 to 132.381537, 10 fc rgb "#00FF00"
set object 17 rect from 0.136980, 0 to 133.056367, 10 fc rgb "#0000FF"
set object 18 rect from 0.137377, 0 to 133.900874, 10 fc rgb "#FFFF00"
set object 19 rect from 0.138277, 0 to 135.231131, 10 fc rgb "#FF00FF"
set object 20 rect from 0.139609, 0 to 137.464077, 10 fc rgb "#808080"
set object 21 rect from 0.141927, 0 to 138.108842, 10 fc rgb "#800080"
set object 22 rect from 0.142895, 0 to 138.744880, 10 fc rgb "#008080"
set object 23 rect from 0.143248, 0 to 156.112970, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.159591, 10 to 160.105709, 20 fc rgb "#FF0000"
set object 25 rect from 0.114585, 10 to 113.713255, 20 fc rgb "#00FF00"
set object 26 rect from 0.117752, 10 to 114.512192, 20 fc rgb "#0000FF"
set object 27 rect from 0.118240, 10 to 115.398381, 20 fc rgb "#FFFF00"
set object 28 rect from 0.119204, 10 to 116.934203, 20 fc rgb "#FF00FF"
set object 29 rect from 0.120765, 10 to 119.298999, 20 fc rgb "#808080"
set object 30 rect from 0.123213, 10 to 120.031029, 20 fc rgb "#800080"
set object 31 rect from 0.124287, 10 to 120.769849, 20 fc rgb "#008080"
set object 32 rect from 0.124746, 10 to 153.957595, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.157397, 20 to 159.202024, 30 fc rgb "#FF0000"
set object 34 rect from 0.094631, 20 to 93.983286, 30 fc rgb "#00FF00"
set object 35 rect from 0.097415, 20 to 94.710461, 30 fc rgb "#0000FF"
set object 36 rect from 0.097823, 20 to 97.057819, 30 fc rgb "#FFFF00"
set object 37 rect from 0.100271, 20 to 98.600417, 30 fc rgb "#FF00FF"
set object 38 rect from 0.101838, 20 to 100.732517, 30 fc rgb "#808080"
set object 39 rect from 0.104050, 20 to 101.481032, 30 fc rgb "#800080"
set object 40 rect from 0.105142, 20 to 102.294518, 30 fc rgb "#008080"
set object 41 rect from 0.105657, 20 to 152.061097, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.155457, 30 to 157.386029, 40 fc rgb "#FF0000"
set object 43 rect from 0.072406, 30 to 72.658346, 40 fc rgb "#00FF00"
set object 44 rect from 0.075434, 30 to 73.454379, 40 fc rgb "#0000FF"
set object 45 rect from 0.075903, 30 to 75.806578, 40 fc rgb "#FFFF00"
set object 46 rect from 0.078322, 30 to 77.372451, 40 fc rgb "#FF00FF"
set object 47 rect from 0.079947, 30 to 79.475467, 40 fc rgb "#808080"
set object 48 rect from 0.082126, 30 to 80.221063, 40 fc rgb "#800080"
set object 49 rect from 0.083211, 30 to 80.992854, 40 fc rgb "#008080"
set object 50 rect from 0.083707, 30 to 150.016248, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.153394, 40 to 155.665997, 50 fc rgb "#FF0000"
set object 52 rect from 0.050612, 40 to 51.727066, 50 fc rgb "#00FF00"
set object 53 rect from 0.053810, 40 to 52.432916, 50 fc rgb "#0000FF"
set object 54 rect from 0.054225, 40 to 54.912140, 50 fc rgb "#FFFF00"
set object 55 rect from 0.056828, 40 to 56.544907, 50 fc rgb "#FF00FF"
set object 56 rect from 0.058505, 40 to 58.845715, 50 fc rgb "#808080"
set object 57 rect from 0.060872, 40 to 59.657265, 50 fc rgb "#800080"
set object 58 rect from 0.062056, 40 to 60.565761, 50 fc rgb "#008080"
set object 59 rect from 0.062638, 40 to 148.015031, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.151178, 50 to 155.897697, 60 fc rgb "#FF0000"
set object 61 rect from 0.022744, 50 to 25.629868, 60 fc rgb "#00FF00"
set object 62 rect from 0.026972, 50 to 26.923284, 60 fc rgb "#0000FF"
set object 63 rect from 0.027932, 50 to 30.535942, 60 fc rgb "#FFFF00"
set object 64 rect from 0.031718, 50 to 32.669978, 60 fc rgb "#FF00FF"
set object 65 rect from 0.033880, 50 to 35.125926, 60 fc rgb "#808080"
set object 66 rect from 0.036429, 50 to 36.081927, 60 fc rgb "#800080"
set object 67 rect from 0.037953, 50 to 37.574116, 60 fc rgb "#008080"
set object 68 rect from 0.038915, 50 to 145.311851, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
