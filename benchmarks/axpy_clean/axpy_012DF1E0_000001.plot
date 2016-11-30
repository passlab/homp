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

set object 15 rect from 0.180329, 0 to 161.489723, 10 fc rgb "#FF0000"
set object 16 rect from 0.151700, 0 to 130.326216, 10 fc rgb "#00FF00"
set object 17 rect from 0.154043, 0 to 130.850272, 10 fc rgb "#0000FF"
set object 18 rect from 0.154435, 0 to 131.533749, 10 fc rgb "#FFFF00"
set object 19 rect from 0.155253, 0 to 132.468229, 10 fc rgb "#FF00FF"
set object 20 rect from 0.156353, 0 to 139.067262, 10 fc rgb "#808080"
set object 21 rect from 0.164134, 0 to 139.563333, 10 fc rgb "#800080"
set object 22 rect from 0.164988, 0 to 140.073822, 10 fc rgb "#008080"
set object 23 rect from 0.165321, 0 to 152.451882, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.173805, 10 to 156.487449, 20 fc rgb "#FF0000"
set object 25 rect from 0.051419, 10 to 45.590305, 20 fc rgb "#00FF00"
set object 26 rect from 0.054120, 10 to 46.289893, 20 fc rgb "#0000FF"
set object 27 rect from 0.054722, 10 to 47.105657, 20 fc rgb "#FFFF00"
set object 28 rect from 0.055697, 10 to 48.152073, 20 fc rgb "#FF00FF"
set object 29 rect from 0.056948, 10 to 54.850318, 20 fc rgb "#808080"
set object 30 rect from 0.064844, 10 to 55.399813, 20 fc rgb "#800080"
set object 31 rect from 0.065745, 10 to 55.930654, 20 fc rgb "#008080"
set object 32 rect from 0.066088, 10 to 146.812772, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.171909, 20 to 161.164947, 30 fc rgb "#FF0000"
set object 34 rect from 0.018722, 20 to 18.492044, 30 fc rgb "#00FF00"
set object 35 rect from 0.022368, 20 to 19.852215, 30 fc rgb "#0000FF"
set object 36 rect from 0.023542, 20 to 23.690203, 30 fc rgb "#FFFF00"
set object 37 rect from 0.028096, 20 to 27.289906, 30 fc rgb "#FF00FF"
set object 38 rect from 0.032313, 20 to 34.019527, 30 fc rgb "#808080"
set object 39 rect from 0.040335, 20 to 34.773387, 30 fc rgb "#800080"
set object 40 rect from 0.041750, 20 to 35.939369, 30 fc rgb "#008080"
set object 41 rect from 0.042515, 20 to 144.976032, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.175620, 30 to 161.729706, 40 fc rgb "#FF0000"
set object 43 rect from 0.073489, 30 to 63.992462, 40 fc rgb "#00FF00"
set object 44 rect from 0.075952, 30 to 64.670003, 40 fc rgb "#0000FF"
set object 45 rect from 0.076389, 30 to 67.331833, 40 fc rgb "#FFFF00"
set object 46 rect from 0.079534, 30 to 70.279434, 40 fc rgb "#FF00FF"
set object 47 rect from 0.083019, 30 to 76.931891, 40 fc rgb "#808080"
set object 48 rect from 0.090860, 30 to 77.562793, 40 fc rgb "#800080"
set object 49 rect from 0.091909, 30 to 78.276797, 40 fc rgb "#008080"
set object 50 rect from 0.092438, 30 to 148.269613, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.178745, 40 to 163.772502, 50 fc rgb "#FF0000"
set object 52 rect from 0.126153, 40 to 108.412545, 50 fc rgb "#00FF00"
set object 53 rect from 0.128209, 40 to 109.028184, 50 fc rgb "#0000FF"
set object 54 rect from 0.128686, 40 to 111.572991, 50 fc rgb "#FFFF00"
set object 55 rect from 0.131693, 40 to 114.212773, 50 fc rgb "#FF00FF"
set object 56 rect from 0.134804, 40 to 120.685456, 50 fc rgb "#808080"
set object 57 rect from 0.142434, 40 to 121.215447, 50 fc rgb "#800080"
set object 58 rect from 0.143347, 40 to 121.770031, 50 fc rgb "#008080"
set object 59 rect from 0.143717, 40 to 151.164638, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.177221, 50 to 163.292545, 60 fc rgb "#FF0000"
set object 61 rect from 0.100490, 50 to 86.677800, 60 fc rgb "#00FF00"
set object 62 rect from 0.102573, 50 to 87.331599, 60 fc rgb "#0000FF"
set object 63 rect from 0.103099, 50 to 89.993428, 60 fc rgb "#FFFF00"
set object 64 rect from 0.106243, 50 to 93.114867, 60 fc rgb "#FF00FF"
set object 65 rect from 0.109930, 50 to 99.724074, 60 fc rgb "#808080"
set object 66 rect from 0.117733, 50 to 100.304946, 60 fc rgb "#800080"
set object 67 rect from 0.118703, 50 to 100.917191, 60 fc rgb "#008080"
set object 68 rect from 0.119210, 50 to 149.840084, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
