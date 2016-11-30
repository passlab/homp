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

set object 15 rect from 0.343217, 0 to 157.165919, 10 fc rgb "#FF0000"
set object 16 rect from 0.042253, 0 to 19.645582, 10 fc rgb "#00FF00"
set object 17 rect from 0.046947, 0 to 20.241709, 10 fc rgb "#0000FF"
set object 18 rect from 0.048042, 0 to 21.444103, 10 fc rgb "#FFFF00"
set object 19 rect from 0.050916, 0 to 22.150076, 10 fc rgb "#FF00FF"
set object 20 rect from 0.052561, 0 to 31.882878, 10 fc rgb "#808080"
set object 21 rect from 0.075628, 0 to 32.186645, 10 fc rgb "#800080"
set object 22 rect from 0.076770, 0 to 32.545758, 10 fc rgb "#008080"
set object 23 rect from 0.077196, 0 to 144.579228, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.347723, 10 to 157.435888, 20 fc rgb "#FF0000"
set object 25 rect from 0.142533, 10 to 61.077902, 20 fc rgb "#00FF00"
set object 26 rect from 0.144921, 10 to 61.376176, 20 fc rgb "#0000FF"
set object 27 rect from 0.145388, 10 to 61.694731, 20 fc rgb "#FFFF00"
set object 28 rect from 0.146167, 10 to 62.168337, 20 fc rgb "#FF00FF"
set object 29 rect from 0.147288, 10 to 71.663279, 20 fc rgb "#808080"
set object 30 rect from 0.169775, 10 to 71.937472, 20 fc rgb "#800080"
set object 31 rect from 0.170686, 10 to 72.194766, 20 fc rgb "#008080"
set object 32 rect from 0.171004, 10 to 146.487173, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.351280, 20 to 167.157704, 30 fc rgb "#FF0000"
set object 34 rect from 0.235026, 20 to 100.062340, 30 fc rgb "#00FF00"
set object 35 rect from 0.237193, 20 to 100.376670, 30 fc rgb "#0000FF"
set object 36 rect from 0.237701, 20 to 103.679662, 30 fc rgb "#FFFF00"
set object 37 rect from 0.245570, 20 to 109.126769, 30 fc rgb "#FF00FF"
set object 38 rect from 0.258417, 20 to 118.763244, 30 fc rgb "#808080"
set object 39 rect from 0.281230, 20 to 119.132074, 30 fc rgb "#800080"
set object 40 rect from 0.282352, 20 to 119.398662, 30 fc rgb "#008080"
set object 41 rect from 0.282738, 20 to 148.195282, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.349623, 30 to 164.578006, 40 fc rgb "#FF0000"
set object 43 rect from 0.181075, 30 to 77.352471, 40 fc rgb "#00FF00"
set object 44 rect from 0.183461, 30 to 77.665110, 40 fc rgb "#0000FF"
set object 45 rect from 0.183941, 30 to 79.129445, 40 fc rgb "#FFFF00"
set object 46 rect from 0.187436, 30 to 84.667809, 40 fc rgb "#FF00FF"
set object 47 rect from 0.200527, 30 to 94.153456, 40 fc rgb "#808080"
set object 48 rect from 0.222976, 30 to 94.537496, 40 fc rgb "#800080"
set object 49 rect from 0.224159, 30 to 94.842530, 40 fc rgb "#008080"
set object 50 rect from 0.224623, 30 to 147.431006, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.352790, 40 to 165.032601, 50 fc rgb "#FF0000"
set object 52 rect from 0.293322, 40 to 124.683957, 50 fc rgb "#00FF00"
set object 53 rect from 0.295471, 40 to 124.929422, 50 fc rgb "#0000FF"
set object 54 rect from 0.295815, 40 to 126.182091, 50 fc rgb "#FFFF00"
set object 55 rect from 0.298783, 40 to 131.111654, 50 fc rgb "#FF00FF"
set object 56 rect from 0.310449, 40 to 140.604483, 50 fc rgb "#808080"
set object 57 rect from 0.332922, 40 to 140.964863, 50 fc rgb "#800080"
set object 58 rect from 0.334035, 40 to 141.229761, 50 fc rgb "#008080"
set object 59 rect from 0.334407, 40 to 148.849712, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.345346, 50 to 163.076071, 60 fc rgb "#FF0000"
set object 61 rect from 0.087804, 50 to 38.091727, 60 fc rgb "#00FF00"
set object 62 rect from 0.090523, 50 to 38.463091, 60 fc rgb "#0000FF"
set object 63 rect from 0.091156, 50 to 39.948128, 60 fc rgb "#FFFF00"
set object 64 rect from 0.094711, 50 to 45.636052, 60 fc rgb "#FF00FF"
set object 65 rect from 0.108165, 50 to 55.154231, 60 fc rgb "#808080"
set object 66 rect from 0.130730, 50 to 55.614318, 60 fc rgb "#800080"
set object 67 rect from 0.132031, 50 to 56.042718, 60 fc rgb "#008080"
set object 68 rect from 0.132793, 50 to 145.604178, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
