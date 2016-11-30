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

set object 15 rect from 0.292596, 0 to 157.167973, 10 fc rgb "#FF0000"
set object 16 rect from 0.033794, 0 to 18.987424, 10 fc rgb "#00FF00"
set object 17 rect from 0.038826, 0 to 19.651949, 10 fc rgb "#0000FF"
set object 18 rect from 0.039736, 0 to 20.459410, 10 fc rgb "#FFFF00"
set object 19 rect from 0.041405, 0 to 21.295650, 10 fc rgb "#FF00FF"
set object 20 rect from 0.043052, 0 to 31.070015, 10 fc rgb "#808080"
set object 21 rect from 0.062772, 0 to 31.467043, 10 fc rgb "#800080"
set object 22 rect from 0.063997, 0 to 31.884915, 10 fc rgb "#008080"
set object 23 rect from 0.064410, 0 to 144.624824, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.294529, 10 to 157.167470, 20 fc rgb "#FF0000"
set object 25 rect from 0.122041, 10 to 61.659623, 20 fc rgb "#00FF00"
set object 26 rect from 0.124637, 10 to 62.032828, 20 fc rgb "#0000FF"
set object 27 rect from 0.125113, 10 to 62.439785, 20 fc rgb "#FFFF00"
set object 28 rect from 0.125956, 10 to 63.035823, 20 fc rgb "#FF00FF"
set object 29 rect from 0.127157, 10 to 72.763039, 20 fc rgb "#808080"
set object 30 rect from 0.146750, 10 to 73.043937, 20 fc rgb "#800080"
set object 31 rect from 0.147601, 10 to 73.341710, 20 fc rgb "#008080"
set object 32 rect from 0.147926, 10 to 145.840724, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.301772, 20 to 165.730893, 30 fc rgb "#FF0000"
set object 34 rect from 0.248716, 20 to 124.386262, 30 fc rgb "#00FF00"
set object 35 rect from 0.251011, 20 to 124.726218, 30 fc rgb "#0000FF"
set object 36 rect from 0.251441, 20 to 126.308378, 30 fc rgb "#FFFF00"
set object 37 rect from 0.254629, 20 to 130.507953, 30 fc rgb "#FF00FF"
set object 38 rect from 0.263096, 20 to 140.241622, 30 fc rgb "#808080"
set object 39 rect from 0.282712, 20 to 140.725996, 30 fc rgb "#800080"
set object 40 rect from 0.283988, 20 to 141.110619, 30 fc rgb "#008080"
set object 41 rect from 0.284461, 20 to 149.521186, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.296671, 30 to 163.978505, 40 fc rgb "#FF0000"
set object 43 rect from 0.074802, 30 to 38.339122, 40 fc rgb "#00FF00"
set object 44 rect from 0.077615, 30 to 38.695455, 40 fc rgb "#0000FF"
set object 45 rect from 0.078091, 30 to 40.512855, 40 fc rgb "#FFFF00"
set object 46 rect from 0.081788, 30 to 45.188868, 40 fc rgb "#FF00FF"
set object 47 rect from 0.091205, 30 to 54.978615, 40 fc rgb "#808080"
set object 48 rect from 0.110954, 30 to 55.504180, 40 fc rgb "#800080"
set object 49 rect from 0.112298, 30 to 56.258040, 40 fc rgb "#008080"
set object 50 rect from 0.113515, 30 to 146.867543, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.298457, 40 to 164.616729, 50 fc rgb "#FF0000"
set object 52 rect from 0.157848, 40 to 79.359679, 50 fc rgb "#00FF00"
set object 53 rect from 0.160357, 40 to 79.781522, 50 fc rgb "#0000FF"
set object 54 rect from 0.160876, 40 to 81.434652, 50 fc rgb "#FFFF00"
set object 55 rect from 0.164237, 40 to 86.058059, 50 fc rgb "#FF00FF"
set object 56 rect from 0.173547, 40 to 95.788248, 50 fc rgb "#808080"
set object 57 rect from 0.193151, 40 to 96.294464, 50 fc rgb "#800080"
set object 58 rect from 0.194470, 40 to 96.722260, 50 fc rgb "#008080"
set object 59 rect from 0.195017, 40 to 147.837285, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.300120, 50 to 165.214749, 60 fc rgb "#FF0000"
set object 61 rect from 0.203134, 50 to 101.790340, 60 fc rgb "#00FF00"
set object 62 rect from 0.205485, 50 to 102.100019, 60 fc rgb "#0000FF"
set object 63 rect from 0.205848, 50 to 103.662827, 60 fc rgb "#FFFF00"
set object 64 rect from 0.209002, 50 to 108.235114, 60 fc rgb "#FF00FF"
set object 65 rect from 0.218214, 50 to 117.928583, 60 fc rgb "#808080"
set object 66 rect from 0.237755, 50 to 118.438267, 60 fc rgb "#800080"
set object 67 rect from 0.239089, 50 to 118.782194, 60 fc rgb "#008080"
set object 68 rect from 0.239468, 50 to 148.692882, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
