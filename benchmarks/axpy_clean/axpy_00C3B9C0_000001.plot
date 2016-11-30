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

set object 15 rect from 0.126821, 0 to 155.465529, 10 fc rgb "#FF0000"
set object 16 rect from 0.062988, 0 to 77.412259, 10 fc rgb "#00FF00"
set object 17 rect from 0.065339, 0 to 78.214119, 10 fc rgb "#0000FF"
set object 18 rect from 0.065761, 0 to 79.157765, 10 fc rgb "#FFFF00"
set object 19 rect from 0.066567, 0 to 80.413575, 10 fc rgb "#FF00FF"
set object 20 rect from 0.067618, 0 to 82.016101, 10 fc rgb "#808080"
set object 21 rect from 0.068968, 0 to 82.674985, 10 fc rgb "#800080"
set object 22 rect from 0.069775, 0 to 83.315995, 10 fc rgb "#008080"
set object 23 rect from 0.070048, 0 to 150.388668, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.125148, 10 to 154.140609, 20 fc rgb "#FF0000"
set object 25 rect from 0.047055, 10 to 58.894416, 20 fc rgb "#00FF00"
set object 26 rect from 0.049805, 10 to 59.903592, 20 fc rgb "#0000FF"
set object 27 rect from 0.050392, 10 to 60.934212, 20 fc rgb "#FFFF00"
set object 28 rect from 0.051286, 10 to 62.489085, 20 fc rgb "#FF00FF"
set object 29 rect from 0.052599, 10 to 64.182162, 20 fc rgb "#808080"
set object 30 rect from 0.054014, 10 to 64.897047, 20 fc rgb "#800080"
set object 31 rect from 0.054900, 10 to 65.681033, 20 fc rgb "#008080"
set object 32 rect from 0.055274, 10 to 148.271423, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.123002, 20 to 154.644605, 30 fc rgb "#FF0000"
set object 34 rect from 0.025838, 20 to 34.205944, 30 fc rgb "#00FF00"
set object 35 rect from 0.029245, 20 to 35.717919, 30 fc rgb "#0000FF"
set object 36 rect from 0.030121, 20 to 38.485708, 30 fc rgb "#FFFF00"
set object 37 rect from 0.032451, 20 to 40.657760, 30 fc rgb "#FF00FF"
set object 38 rect from 0.034247, 20 to 42.479520, 30 fc rgb "#808080"
set object 39 rect from 0.035784, 20 to 43.394568, 30 fc rgb "#800080"
set object 40 rect from 0.036985, 20 to 44.886289, 30 fc rgb "#008080"
set object 41 rect from 0.037810, 20 to 145.276064, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.128518, 30 to 157.831791, 40 fc rgb "#FF0000"
set object 43 rect from 0.078756, 30 to 96.218437, 40 fc rgb "#00FF00"
set object 44 rect from 0.081112, 30 to 97.061997, 40 fc rgb "#0000FF"
set object 45 rect from 0.081578, 30 to 98.142662, 40 fc rgb "#FFFF00"
set object 46 rect from 0.082507, 30 to 99.546215, 40 fc rgb "#FF00FF"
set object 47 rect from 0.083678, 30 to 101.016493, 40 fc rgb "#808080"
set object 48 rect from 0.084922, 30 to 101.832649, 40 fc rgb "#800080"
set object 49 rect from 0.085875, 30 to 102.741741, 40 fc rgb "#008080"
set object 50 rect from 0.086350, 30 to 152.351020, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.131507, 40 to 161.204851, 50 fc rgb "#FF0000"
set object 52 rect from 0.109504, 40 to 132.517796, 50 fc rgb "#00FF00"
set object 53 rect from 0.111589, 40 to 133.347059, 50 fc rgb "#0000FF"
set object 54 rect from 0.112030, 40 to 134.355042, 50 fc rgb "#FFFF00"
set object 55 rect from 0.112887, 40 to 135.790766, 50 fc rgb "#FF00FF"
set object 56 rect from 0.114083, 40 to 137.188362, 50 fc rgb "#808080"
set object 57 rect from 0.115269, 40 to 137.903244, 50 fc rgb "#800080"
set object 58 rect from 0.116114, 40 to 138.627659, 50 fc rgb "#008080"
set object 59 rect from 0.116470, 40 to 156.139899, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.130016, 50 to 159.676188, 60 fc rgb "#FF0000"
set object 61 rect from 0.094060, 50 to 114.033312, 60 fc rgb "#00FF00"
set object 62 rect from 0.096081, 50 to 114.976958, 60 fc rgb "#0000FF"
set object 63 rect from 0.096618, 50 to 116.052855, 60 fc rgb "#FFFF00"
set object 64 rect from 0.097534, 50 to 117.546960, 60 fc rgb "#FF00FF"
set object 65 rect from 0.098766, 50 to 118.962426, 60 fc rgb "#808080"
set object 66 rect from 0.099960, 50 to 119.666588, 60 fc rgb "#800080"
set object 67 rect from 0.100806, 50 to 120.437467, 60 fc rgb "#008080"
set object 68 rect from 0.101200, 50 to 154.332436, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
