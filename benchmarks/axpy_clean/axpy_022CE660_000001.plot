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

set object 15 rect from 0.153819, 0 to 159.842547, 10 fc rgb "#FF0000"
set object 16 rect from 0.131131, 0 to 133.162633, 10 fc rgb "#00FF00"
set object 17 rect from 0.133038, 0 to 133.746764, 10 fc rgb "#0000FF"
set object 18 rect from 0.133383, 0 to 134.485445, 10 fc rgb "#FFFF00"
set object 19 rect from 0.134134, 0 to 135.544292, 10 fc rgb "#FF00FF"
set object 20 rect from 0.135173, 0 to 138.822189, 10 fc rgb "#808080"
set object 21 rect from 0.138440, 0 to 139.368169, 10 fc rgb "#800080"
set object 22 rect from 0.139244, 0 to 139.906132, 10 fc rgb "#008080"
set object 23 rect from 0.139521, 0 to 153.739366, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.152212, 10 to 158.571915, 20 fc rgb "#FF0000"
set object 25 rect from 0.114458, 10 to 116.545287, 20 fc rgb "#00FF00"
set object 26 rect from 0.116485, 10 to 117.228767, 20 fc rgb "#0000FF"
set object 27 rect from 0.116925, 10 to 117.958415, 20 fc rgb "#FFFF00"
set object 28 rect from 0.117676, 10 to 119.167805, 20 fc rgb "#FF00FF"
set object 29 rect from 0.118883, 10 to 122.493888, 20 fc rgb "#808080"
set object 30 rect from 0.122201, 10 to 123.159302, 20 fc rgb "#800080"
set object 31 rect from 0.123117, 10 to 123.802642, 20 fc rgb "#008080"
set object 32 rect from 0.123478, 10 to 152.094398, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.150578, 20 to 158.382234, 30 fc rgb "#FF0000"
set object 34 rect from 0.095481, 20 to 97.539253, 30 fc rgb "#00FF00"
set object 35 rect from 0.097624, 20 to 98.302027, 30 fc rgb "#0000FF"
set object 36 rect from 0.098066, 20 to 100.273189, 30 fc rgb "#FFFF00"
set object 37 rect from 0.100063, 20 to 101.751553, 30 fc rgb "#FF00FF"
set object 38 rect from 0.101503, 20 to 105.015406, 30 fc rgb "#808080"
set object 39 rect from 0.104759, 20 to 105.637674, 30 fc rgb "#800080"
set object 40 rect from 0.105670, 20 to 106.503820, 30 fc rgb "#008080"
set object 41 rect from 0.106240, 20 to 150.625052, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.149053, 30 to 156.532511, 40 fc rgb "#FF0000"
set object 43 rect from 0.053637, 30 to 55.372103, 40 fc rgb "#00FF00"
set object 44 rect from 0.055539, 30 to 55.973299, 40 fc rgb "#0000FF"
set object 45 rect from 0.055890, 30 to 57.901299, 40 fc rgb "#FFFF00"
set object 46 rect from 0.057812, 30 to 59.206030, 40 fc rgb "#FF00FF"
set object 47 rect from 0.059112, 30 to 62.416687, 40 fc rgb "#808080"
set object 48 rect from 0.062321, 30 to 63.051996, 40 fc rgb "#800080"
set object 49 rect from 0.063213, 30 to 63.698356, 40 fc rgb "#008080"
set object 50 rect from 0.063592, 30 to 149.049328, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.147376, 40 to 154.686826, 50 fc rgb "#FF0000"
set object 52 rect from 0.036774, 40 to 38.752754, 50 fc rgb "#00FF00"
set object 53 rect from 0.038977, 40 to 39.392071, 50 fc rgb "#0000FF"
set object 54 rect from 0.039368, 40 to 41.087229, 50 fc rgb "#FFFF00"
set object 55 rect from 0.041065, 40 to 42.430111, 50 fc rgb "#FF00FF"
set object 56 rect from 0.042408, 40 to 45.629731, 50 fc rgb "#808080"
set object 57 rect from 0.045587, 40 to 46.276077, 50 fc rgb "#800080"
set object 58 rect from 0.046503, 40 to 47.002719, 50 fc rgb "#008080"
set object 59 rect from 0.046976, 40 to 147.272887, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.145505, 50 to 155.292014, 60 fc rgb "#FF0000"
set object 61 rect from 0.015444, 50 to 18.018449, 60 fc rgb "#00FF00"
set object 62 rect from 0.018505, 50 to 19.234868, 60 fc rgb "#0000FF"
set object 63 rect from 0.019288, 50 to 22.302999, 60 fc rgb "#FFFF00"
set object 64 rect from 0.022385, 50 to 24.078454, 60 fc rgb "#FF00FF"
set object 65 rect from 0.024125, 50 to 27.496872, 60 fc rgb "#808080"
set object 66 rect from 0.027549, 50 to 28.269682, 60 fc rgb "#800080"
set object 67 rect from 0.028714, 50 to 29.481075, 60 fc rgb "#008080"
set object 68 rect from 0.029505, 50 to 145.157195, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
