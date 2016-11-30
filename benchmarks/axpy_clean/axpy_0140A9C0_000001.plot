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

set object 15 rect from 0.103597, 0 to 160.792768, 10 fc rgb "#FF0000"
set object 16 rect from 0.086429, 0 to 132.608694, 10 fc rgb "#00FF00"
set object 17 rect from 0.087991, 0 to 133.460280, 10 fc rgb "#0000FF"
set object 18 rect from 0.088334, 0 to 134.373884, 10 fc rgb "#FFFF00"
set object 19 rect from 0.088966, 0 to 135.701935, 10 fc rgb "#FF00FF"
set object 20 rect from 0.089816, 0 to 137.039065, 10 fc rgb "#808080"
set object 21 rect from 0.090711, 0 to 137.706116, 10 fc rgb "#800080"
set object 22 rect from 0.091359, 0 to 138.400395, 10 fc rgb "#008080"
set object 23 rect from 0.091602, 0 to 155.978191, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.102323, 10 to 159.370936, 20 fc rgb "#FF0000"
set object 25 rect from 0.074033, 10 to 114.065859, 20 fc rgb "#00FF00"
set object 26 rect from 0.075732, 10 to 115.114085, 20 fc rgb "#0000FF"
set object 27 rect from 0.076205, 10 to 116.023152, 20 fc rgb "#FFFF00"
set object 28 rect from 0.076856, 10 to 117.579606, 20 fc rgb "#FF00FF"
set object 29 rect from 0.077843, 10 to 118.919761, 20 fc rgb "#808080"
set object 30 rect from 0.078747, 10 to 119.712357, 20 fc rgb "#800080"
set object 31 rect from 0.079463, 10 to 120.485287, 20 fc rgb "#008080"
set object 32 rect from 0.079757, 10 to 153.936195, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.100891, 20 to 156.729951, 30 fc rgb "#FF0000"
set object 34 rect from 0.063029, 20 to 97.043184, 30 fc rgb "#00FF00"
set object 35 rect from 0.064487, 20 to 97.873595, 30 fc rgb "#0000FF"
set object 36 rect from 0.064806, 20 to 98.871902, 30 fc rgb "#FFFF00"
set object 37 rect from 0.065474, 20 to 100.343653, 30 fc rgb "#FF00FF"
set object 38 rect from 0.066451, 20 to 101.476581, 30 fc rgb "#808080"
set object 39 rect from 0.067192, 20 to 102.158757, 30 fc rgb "#800080"
set object 40 rect from 0.067873, 20 to 102.984632, 30 fc rgb "#008080"
set object 41 rect from 0.068186, 20 to 152.068152, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.099684, 30 to 155.137187, 40 fc rgb "#FF0000"
set object 43 rect from 0.050793, 30 to 78.725726, 40 fc rgb "#00FF00"
set object 44 rect from 0.052362, 30 to 79.580339, 40 fc rgb "#0000FF"
set object 45 rect from 0.052712, 30 to 80.693607, 40 fc rgb "#FFFF00"
set object 46 rect from 0.053479, 30 to 82.154765, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054413, 30 to 83.310385, 40 fc rgb "#808080"
set object 48 rect from 0.055201, 30 to 84.074243, 40 fc rgb "#800080"
set object 49 rect from 0.055914, 30 to 84.915242, 40 fc rgb "#008080"
set object 50 rect from 0.056240, 30 to 150.138086, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.098417, 40 to 153.331166, 50 fc rgb "#FF0000"
set object 52 rect from 0.038420, 40 to 60.370458, 50 fc rgb "#00FF00"
set object 53 rect from 0.040216, 40 to 61.167596, 50 fc rgb "#0000FF"
set object 54 rect from 0.040541, 40 to 62.295982, 50 fc rgb "#FFFF00"
set object 55 rect from 0.041314, 40 to 63.829750, 50 fc rgb "#FF00FF"
set object 56 rect from 0.042314, 40 to 65.038310, 50 fc rgb "#808080"
set object 57 rect from 0.043125, 40 to 65.839983, 50 fc rgb "#800080"
set object 58 rect from 0.043868, 40 to 66.706694, 50 fc rgb "#008080"
set object 59 rect from 0.044221, 40 to 148.047687, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.096934, 50 to 153.612502, 60 fc rgb "#FF0000"
set object 61 rect from 0.022289, 50 to 37.223305, 60 fc rgb "#00FF00"
set object 62 rect from 0.024991, 50 to 38.567997, 60 fc rgb "#0000FF"
set object 63 rect from 0.025612, 50 to 40.705289, 60 fc rgb "#FFFF00"
set object 64 rect from 0.027058, 50 to 42.916691, 60 fc rgb "#FF00FF"
set object 65 rect from 0.028479, 50 to 44.370292, 60 fc rgb "#808080"
set object 66 rect from 0.029470, 50 to 45.380702, 60 fc rgb "#800080"
set object 67 rect from 0.030610, 50 to 47.043034, 60 fc rgb "#008080"
set object 68 rect from 0.031225, 50 to 145.264524, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
