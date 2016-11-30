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

set object 15 rect from 0.282698, 0 to 159.286668, 10 fc rgb "#FF0000"
set object 16 rect from 0.169793, 0 to 90.074002, 10 fc rgb "#00FF00"
set object 17 rect from 0.172135, 0 to 90.422189, 10 fc rgb "#0000FF"
set object 18 rect from 0.172547, 0 to 90.837496, 10 fc rgb "#FFFF00"
set object 19 rect from 0.173372, 0 to 91.477237, 10 fc rgb "#FF00FF"
set object 20 rect from 0.174597, 0 to 101.198160, 10 fc rgb "#808080"
set object 21 rect from 0.193120, 0 to 101.530615, 10 fc rgb "#800080"
set object 22 rect from 0.194015, 0 to 101.856779, 10 fc rgb "#008080"
set object 23 rect from 0.194380, 0 to 147.908713, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.285855, 10 to 160.867670, 20 fc rgb "#FF0000"
set object 25 rect from 0.245225, 10 to 129.514579, 20 fc rgb "#00FF00"
set object 26 rect from 0.247332, 10 to 129.898948, 20 fc rgb "#0000FF"
set object 27 rect from 0.247835, 10 to 130.329461, 20 fc rgb "#FFFF00"
set object 28 rect from 0.248665, 10 to 130.928826, 20 fc rgb "#FF00FF"
set object 29 rect from 0.249796, 10 to 140.544871, 20 fc rgb "#808080"
set object 30 rect from 0.268176, 10 to 140.874182, 20 fc rgb "#800080"
set object 31 rect from 0.269031, 10 to 141.176224, 20 fc rgb "#008080"
set object 32 rect from 0.269345, 10 to 149.597212, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.280937, 20 to 165.080527, 30 fc rgb "#FF0000"
set object 34 rect from 0.122730, 20 to 65.346951, 30 fc rgb "#00FF00"
set object 35 rect from 0.124982, 20 to 65.701432, 30 fc rgb "#0000FF"
set object 36 rect from 0.125426, 20 to 67.303407, 30 fc rgb "#FFFF00"
set object 37 rect from 0.128465, 20 to 73.329036, 30 fc rgb "#FF00FF"
set object 38 rect from 0.139967, 20 to 83.000141, 30 fc rgb "#808080"
set object 39 rect from 0.158429, 20 to 83.513507, 30 fc rgb "#800080"
set object 40 rect from 0.159661, 20 to 83.959754, 30 fc rgb "#008080"
set object 41 rect from 0.160230, 20 to 146.970077, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.284277, 30 to 165.080000, 40 fc rgb "#FF0000"
set object 43 rect from 0.202289, 30 to 106.966842, 40 fc rgb "#00FF00"
set object 44 rect from 0.204325, 30 to 107.386870, 40 fc rgb "#0000FF"
set object 45 rect from 0.204901, 30 to 109.012968, 40 fc rgb "#FFFF00"
set object 46 rect from 0.208022, 30 to 113.321257, 40 fc rgb "#FF00FF"
set object 47 rect from 0.216237, 30 to 122.881718, 40 fc rgb "#808080"
set object 48 rect from 0.234489, 30 to 123.372537, 40 fc rgb "#800080"
set object 49 rect from 0.235676, 30 to 123.815637, 40 fc rgb "#008080"
set object 50 rect from 0.236239, 30 to 148.785998, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.279162, 40 to 162.739697, 50 fc rgb "#FF0000"
set object 52 rect from 0.079431, 40 to 42.847460, 50 fc rgb "#00FF00"
set object 53 rect from 0.082078, 40 to 43.284791, 50 fc rgb "#0000FF"
set object 54 rect from 0.082654, 40 to 44.995837, 50 fc rgb "#FFFF00"
set object 55 rect from 0.085948, 40 to 49.416868, 50 fc rgb "#FF00FF"
set object 56 rect from 0.094356, 40 to 59.052840, 50 fc rgb "#808080"
set object 57 rect from 0.112753, 40 to 59.584561, 50 fc rgb "#800080"
set object 58 rect from 0.114039, 40 to 59.964210, 50 fc rgb "#008080"
set object 59 rect from 0.114514, 40 to 146.062378, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.277259, 50 to 162.949977, 60 fc rgb "#FF0000"
set object 61 rect from 0.029728, 50 to 17.294513, 60 fc rgb "#00FF00"
set object 62 rect from 0.033477, 50 to 17.924817, 60 fc rgb "#0000FF"
set object 63 rect from 0.034322, 50 to 20.118290, 60 fc rgb "#FFFF00"
set object 64 rect from 0.038551, 50 to 25.034859, 60 fc rgb "#FF00FF"
set object 65 rect from 0.047870, 50 to 34.834438, 60 fc rgb "#808080"
set object 66 rect from 0.066592, 50 to 35.371925, 60 fc rgb "#800080"
set object 67 rect from 0.068152, 50 to 36.121263, 60 fc rgb "#008080"
set object 68 rect from 0.069022, 50 to 144.620863, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
