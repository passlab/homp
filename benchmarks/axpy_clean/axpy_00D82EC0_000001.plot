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

set object 15 rect from 0.276537, 0 to 159.043912, 10 fc rgb "#FF0000"
set object 16 rect from 0.241102, 0 to 131.379513, 10 fc rgb "#00FF00"
set object 17 rect from 0.242794, 0 to 131.701881, 10 fc rgb "#0000FF"
set object 18 rect from 0.243184, 0 to 132.023699, 10 fc rgb "#FFFF00"
set object 19 rect from 0.243791, 0 to 132.493978, 10 fc rgb "#FF00FF"
set object 20 rect from 0.244663, 0 to 140.697778, 10 fc rgb "#808080"
set object 21 rect from 0.259803, 0 to 140.947002, 10 fc rgb "#800080"
set object 22 rect from 0.260487, 0 to 141.211944, 10 fc rgb "#008080"
set object 23 rect from 0.260739, 0 to 149.574490, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.269807, 10 to 155.806721, 20 fc rgb "#FF0000"
set object 25 rect from 0.088137, 10 to 48.623058, 20 fc rgb "#00FF00"
set object 26 rect from 0.090088, 10 to 49.056499, 20 fc rgb "#0000FF"
set object 27 rect from 0.090643, 10 to 49.438456, 20 fc rgb "#FFFF00"
set object 28 rect from 0.091395, 10 to 50.014382, 20 fc rgb "#FF00FF"
set object 29 rect from 0.092433, 10 to 58.372601, 20 fc rgb "#808080"
set object 30 rect from 0.107864, 10 to 58.648369, 20 fc rgb "#800080"
set object 31 rect from 0.108576, 10 to 58.921433, 20 fc rgb "#008080"
set object 32 rect from 0.108873, 10 to 145.688220, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.272102, 20 to 163.535374, 30 fc rgb "#FF0000"
set object 34 rect from 0.117131, 20 to 64.205525, 30 fc rgb "#00FF00"
set object 35 rect from 0.118844, 20 to 64.506757, 30 fc rgb "#0000FF"
set object 36 rect from 0.119159, 20 to 65.993434, 30 fc rgb "#FFFF00"
set object 37 rect from 0.121931, 20 to 72.218617, 30 fc rgb "#FF00FF"
set object 38 rect from 0.133408, 20 to 80.182947, 30 fc rgb "#808080"
set object 39 rect from 0.148104, 20 to 80.688442, 30 fc rgb "#800080"
set object 40 rect from 0.149299, 20 to 81.041682, 30 fc rgb "#008080"
set object 41 rect from 0.149681, 20 to 146.877452, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.273728, 30 to 164.156244, 40 fc rgb "#FF0000"
set object 43 rect from 0.157527, 30 to 86.022911, 40 fc rgb "#00FF00"
set object 44 rect from 0.159117, 30 to 86.351237, 40 fc rgb "#0000FF"
set object 45 rect from 0.159478, 30 to 87.802699, 40 fc rgb "#FFFF00"
set object 46 rect from 0.162157, 30 to 93.751024, 40 fc rgb "#FF00FF"
set object 47 rect from 0.173139, 30 to 101.734317, 40 fc rgb "#808080"
set object 48 rect from 0.187873, 30 to 102.221389, 40 fc rgb "#800080"
set object 49 rect from 0.188984, 30 to 102.498247, 40 fc rgb "#008080"
set object 50 rect from 0.189301, 30 to 148.009800, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.275196, 40 to 167.915747, 50 fc rgb "#FF0000"
set object 52 rect from 0.196140, 40 to 106.872129, 50 fc rgb "#00FF00"
set object 53 rect from 0.197550, 40 to 107.154397, 50 fc rgb "#0000FF"
set object 54 rect from 0.197875, 40 to 111.445390, 50 fc rgb "#FFFF00"
set object 55 rect from 0.205856, 40 to 117.477693, 50 fc rgb "#FF00FF"
set object 56 rect from 0.216929, 40 to 125.576928, 50 fc rgb "#808080"
set object 57 rect from 0.231879, 40 to 126.038536, 50 fc rgb "#800080"
set object 58 rect from 0.232962, 40 to 126.349520, 50 fc rgb "#008080"
set object 59 rect from 0.233307, 40 to 148.786187, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.267838, 50 to 162.767109, 60 fc rgb "#FF0000"
set object 61 rect from 0.040899, 50 to 23.706624, 60 fc rgb "#00FF00"
set object 62 rect from 0.044287, 50 to 24.374650, 60 fc rgb "#0000FF"
set object 63 rect from 0.045095, 50 to 26.606834, 60 fc rgb "#FFFF00"
set object 64 rect from 0.049242, 50 to 33.134872, 60 fc rgb "#FF00FF"
set object 65 rect from 0.061263, 50 to 41.277460, 60 fc rgb "#808080"
set object 66 rect from 0.076304, 50 to 41.843626, 60 fc rgb "#800080"
set object 67 rect from 0.077728, 50 to 42.452063, 60 fc rgb "#008080"
set object 68 rect from 0.078469, 50 to 144.569960, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
