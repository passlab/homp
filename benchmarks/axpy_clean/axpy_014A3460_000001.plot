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

set object 15 rect from 0.264407, 0 to 155.212543, 10 fc rgb "#FF0000"
set object 16 rect from 0.073570, 0 to 41.728757, 10 fc rgb "#00FF00"
set object 17 rect from 0.075630, 0 to 42.199844, 10 fc rgb "#0000FF"
set object 18 rect from 0.076263, 0 to 42.578931, 10 fc rgb "#FFFF00"
set object 19 rect from 0.076958, 0 to 43.161980, 10 fc rgb "#FF00FF"
set object 20 rect from 0.078028, 0 to 50.533181, 10 fc rgb "#808080"
set object 21 rect from 0.091290, 0 to 50.813065, 10 fc rgb "#800080"
set object 22 rect from 0.092096, 0 to 51.139503, 10 fc rgb "#008080"
set object 23 rect from 0.092398, 0 to 146.161483, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.262218, 10 to 155.141040, 20 fc rgb "#FF0000"
set object 25 rect from 0.039192, 10 to 23.352849, 20 fc rgb "#00FF00"
set object 26 rect from 0.042547, 10 to 24.057820, 20 fc rgb "#0000FF"
set object 27 rect from 0.043529, 10 to 24.863116, 20 fc rgb "#FFFF00"
set object 28 rect from 0.044993, 10 to 25.651219, 20 fc rgb "#FF00FF"
set object 29 rect from 0.046411, 10 to 33.276256, 20 fc rgb "#808080"
set object 30 rect from 0.060169, 10 to 33.608797, 20 fc rgb "#800080"
set object 31 rect from 0.061178, 10 to 34.022248, 20 fc rgb "#008080"
set object 32 rect from 0.061512, 10 to 144.644014, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.268017, 20 to 165.324940, 30 fc rgb "#FF0000"
set object 34 rect from 0.152617, 20 to 85.362377, 30 fc rgb "#00FF00"
set object 35 rect from 0.154348, 20 to 85.696578, 30 fc rgb "#0000FF"
set object 36 rect from 0.154732, 20 to 88.687176, 30 fc rgb "#FFFF00"
set object 37 rect from 0.160220, 20 to 94.806381, 30 fc rgb "#FF00FF"
set object 38 rect from 0.171171, 20 to 102.078933, 30 fc rgb "#808080"
set object 39 rect from 0.184298, 20 to 102.562219, 30 fc rgb "#800080"
set object 40 rect from 0.185406, 20 to 102.895858, 30 fc rgb "#008080"
set object 41 rect from 0.185770, 20 to 148.193827, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.266252, 30 to 166.512643, 40 fc rgb "#FF0000"
set object 43 rect from 0.103597, 30 to 58.258528, 40 fc rgb "#00FF00"
set object 44 rect from 0.105448, 30 to 58.644271, 40 fc rgb "#0000FF"
set object 45 rect from 0.105920, 30 to 61.611596, 40 fc rgb "#FFFF00"
set object 46 rect from 0.111387, 30 to 69.816908, 40 fc rgb "#FF00FF"
set object 47 rect from 0.126090, 30 to 77.123263, 40 fc rgb "#808080"
set object 48 rect from 0.139294, 30 to 77.654762, 40 fc rgb "#800080"
set object 49 rect from 0.140488, 30 to 78.252222, 40 fc rgb "#008080"
set object 50 rect from 0.141317, 30 to 147.162963, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.269436, 40 to 163.416740, 50 fc rgb "#FF0000"
set object 52 rect from 0.192927, 40 to 107.636149, 50 fc rgb "#00FF00"
set object 53 rect from 0.194538, 40 to 107.987527, 50 fc rgb "#0000FF"
set object 54 rect from 0.194954, 40 to 109.485052, 50 fc rgb "#FFFF00"
set object 55 rect from 0.197655, 40 to 114.383850, 50 fc rgb "#FF00FF"
set object 56 rect from 0.206493, 40 to 121.655287, 50 fc rgb "#808080"
set object 57 rect from 0.219632, 40 to 122.098667, 50 fc rgb "#800080"
set object 58 rect from 0.220649, 40 to 122.417903, 50 fc rgb "#008080"
set object 59 rect from 0.220993, 40 to 149.093338, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.270869, 50 to 163.539777, 60 fc rgb "#FF0000"
set object 61 rect from 0.228268, 50 to 127.174265, 60 fc rgb "#00FF00"
set object 62 rect from 0.229792, 50 to 127.507913, 60 fc rgb "#0000FF"
set object 63 rect from 0.230173, 50 to 129.002101, 60 fc rgb "#FFFF00"
set object 64 rect from 0.232889, 50 to 133.263541, 60 fc rgb "#FF00FF"
set object 65 rect from 0.240557, 50 to 140.522789, 60 fc rgb "#808080"
set object 66 rect from 0.253659, 50 to 140.966168, 60 fc rgb "#800080"
set object 67 rect from 0.254698, 50 to 141.292606, 60 fc rgb "#008080"
set object 68 rect from 0.255048, 50 to 149.845416, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
