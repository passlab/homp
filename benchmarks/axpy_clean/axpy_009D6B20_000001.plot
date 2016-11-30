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

set object 15 rect from 0.213373, 0 to 159.337114, 10 fc rgb "#FF0000"
set object 16 rect from 0.184192, 0 to 131.486848, 10 fc rgb "#00FF00"
set object 17 rect from 0.186015, 0 to 131.912433, 10 fc rgb "#0000FF"
set object 18 rect from 0.186409, 0 to 132.433606, 10 fc rgb "#FFFF00"
set object 19 rect from 0.187152, 0 to 133.140301, 10 fc rgb "#FF00FF"
set object 20 rect from 0.188142, 0 to 139.853964, 10 fc rgb "#808080"
set object 21 rect from 0.197634, 0 to 140.246269, 10 fc rgb "#800080"
set object 22 rect from 0.198396, 0 to 140.598909, 10 fc rgb "#008080"
set object 23 rect from 0.198663, 0 to 150.715077, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.210439, 10 to 157.422378, 20 fc rgb "#FF0000"
set object 25 rect from 0.132618, 10 to 94.970628, 20 fc rgb "#00FF00"
set object 26 rect from 0.134466, 10 to 95.455694, 20 fc rgb "#0000FF"
set object 27 rect from 0.134917, 10 to 95.971908, 20 fc rgb "#FFFF00"
set object 28 rect from 0.135667, 10 to 96.777747, 20 fc rgb "#FF00FF"
set object 29 rect from 0.136810, 10 to 103.516196, 20 fc rgb "#808080"
set object 30 rect from 0.146330, 10 to 103.929742, 20 fc rgb "#800080"
set object 31 rect from 0.147119, 10 to 104.332661, 20 fc rgb "#008080"
set object 32 rect from 0.147455, 10 to 148.570903, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.208932, 20 to 161.342511, 30 fc rgb "#FF0000"
set object 34 rect from 0.103461, 20 to 74.319787, 30 fc rgb "#00FF00"
set object 35 rect from 0.105288, 20 to 74.794934, 30 fc rgb "#0000FF"
set object 36 rect from 0.105733, 20 to 77.141638, 30 fc rgb "#FFFF00"
set object 37 rect from 0.109048, 20 to 81.092234, 30 fc rgb "#FF00FF"
set object 38 rect from 0.114649, 20 to 87.686926, 30 fc rgb "#808080"
set object 39 rect from 0.123950, 20 to 88.204565, 30 fc rgb "#800080"
set object 40 rect from 0.124906, 20 to 88.611030, 30 fc rgb "#008080"
set object 41 rect from 0.125248, 20 to 147.564665, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.211932, 30 to 162.988177, 40 fc rgb "#FF0000"
set object 43 rect from 0.154376, 30 to 110.247566, 40 fc rgb "#00FF00"
set object 44 rect from 0.156016, 30 to 110.649778, 40 fc rgb "#0000FF"
set object 45 rect from 0.156367, 30 to 112.798911, 40 fc rgb "#FFFF00"
set object 46 rect from 0.159401, 30 to 116.567510, 40 fc rgb "#FF00FF"
set object 47 rect from 0.164736, 30 to 123.118318, 40 fc rgb "#808080"
set object 48 rect from 0.174000, 30 to 123.648692, 40 fc rgb "#800080"
set object 49 rect from 0.174982, 30 to 124.077812, 40 fc rgb "#008080"
set object 50 rect from 0.175333, 30 to 149.689729, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.207425, 40 to 160.280327, 50 fc rgb "#FF0000"
set object 52 rect from 0.072737, 40 to 52.760438, 50 fc rgb "#00FF00"
set object 53 rect from 0.074835, 40 to 53.211505, 50 fc rgb "#0000FF"
set object 54 rect from 0.075253, 40 to 55.380476, 50 fc rgb "#FFFF00"
set object 55 rect from 0.078341, 40 to 59.505261, 50 fc rgb "#FF00FF"
set object 56 rect from 0.084140, 40 to 66.120497, 50 fc rgb "#808080"
set object 57 rect from 0.093505, 40 to 66.648044, 50 fc rgb "#800080"
set object 58 rect from 0.094511, 40 to 67.150804, 50 fc rgb "#008080"
set object 59 rect from 0.094983, 40 to 146.436627, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.205561, 50 to 161.347470, 60 fc rgb "#FF0000"
set object 61 rect from 0.035105, 50 to 27.371585, 60 fc rgb "#00FF00"
set object 62 rect from 0.039178, 50 to 28.379237, 60 fc rgb "#0000FF"
set object 63 rect from 0.040202, 50 to 31.545931, 60 fc rgb "#FFFF00"
set object 64 rect from 0.044702, 50 to 36.232260, 60 fc rgb "#FF00FF"
set object 65 rect from 0.051321, 50 to 43.163322, 60 fc rgb "#808080"
set object 66 rect from 0.061088, 50 to 43.839564, 60 fc rgb "#800080"
set object 67 rect from 0.062444, 50 to 44.685775, 60 fc rgb "#008080"
set object 68 rect from 0.063236, 50 to 144.787426, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
