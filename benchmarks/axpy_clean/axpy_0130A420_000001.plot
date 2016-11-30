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

set object 15 rect from 0.162331, 0 to 160.174052, 10 fc rgb "#FF0000"
set object 16 rect from 0.138466, 0 to 132.852934, 10 fc rgb "#00FF00"
set object 17 rect from 0.140342, 0 to 133.445016, 10 fc rgb "#0000FF"
set object 18 rect from 0.140747, 0 to 134.050388, 10 fc rgb "#FFFF00"
set object 19 rect from 0.141396, 0 to 134.962223, 10 fc rgb "#FF00FF"
set object 20 rect from 0.142364, 0 to 139.134311, 10 fc rgb "#808080"
set object 21 rect from 0.146771, 0 to 139.672312, 10 fc rgb "#800080"
set object 22 rect from 0.147562, 0 to 140.168560, 10 fc rgb "#008080"
set object 23 rect from 0.147834, 0 to 153.536868, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.155863, 10 to 154.504692, 20 fc rgb "#FF0000"
set object 25 rect from 0.061440, 10 to 60.019373, 20 fc rgb "#00FF00"
set object 26 rect from 0.063588, 10 to 60.693998, 20 fc rgb "#0000FF"
set object 27 rect from 0.064075, 10 to 61.346807, 20 fc rgb "#FFFF00"
set object 28 rect from 0.064805, 10 to 62.457915, 20 fc rgb "#FF00FF"
set object 29 rect from 0.065955, 10 to 66.802683, 20 fc rgb "#808080"
set object 30 rect from 0.070526, 10 to 67.334038, 20 fc rgb "#800080"
set object 31 rect from 0.071350, 10 to 67.898606, 20 fc rgb "#008080"
set object 32 rect from 0.071699, 10 to 147.297242, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.153895, 20 to 158.080896, 30 fc rgb "#FF0000"
set object 34 rect from 0.034457, 20 to 35.310423, 30 fc rgb "#00FF00"
set object 35 rect from 0.037735, 20 to 36.660635, 30 fc rgb "#0000FF"
set object 36 rect from 0.038756, 20 to 40.883949, 30 fc rgb "#FFFF00"
set object 37 rect from 0.043236, 20 to 43.239948, 30 fc rgb "#FF00FF"
set object 38 rect from 0.045695, 20 to 47.560991, 30 fc rgb "#808080"
set object 39 rect from 0.050337, 20 to 48.320082, 30 fc rgb "#800080"
set object 40 rect from 0.051473, 20 to 49.385635, 30 fc rgb "#008080"
set object 41 rect from 0.052173, 20 to 144.986799, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.160883, 30 to 161.130508, 40 fc rgb "#FF0000"
set object 43 rect from 0.119209, 30 to 114.302014, 40 fc rgb "#00FF00"
set object 44 rect from 0.120802, 30 to 114.886503, 40 fc rgb "#0000FF"
set object 45 rect from 0.121212, 30 to 117.652390, 40 fc rgb "#FFFF00"
set object 46 rect from 0.124104, 30 to 119.021576, 40 fc rgb "#FF00FF"
set object 47 rect from 0.125545, 30 to 122.922297, 40 fc rgb "#808080"
set object 48 rect from 0.129664, 30 to 123.436572, 40 fc rgb "#800080"
set object 49 rect from 0.130461, 30 to 123.967927, 40 fc rgb "#008080"
set object 50 rect from 0.130765, 30 to 152.237895, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.157903, 40 to 159.695859, 50 fc rgb "#FF0000"
set object 52 rect from 0.079278, 40 to 76.679238, 50 fc rgb "#00FF00"
set object 53 rect from 0.081136, 40 to 77.462041, 50 fc rgb "#0000FF"
set object 54 rect from 0.081745, 40 to 80.570472, 50 fc rgb "#FFFF00"
set object 55 rect from 0.085032, 40 to 82.654141, 50 fc rgb "#FF00FF"
set object 56 rect from 0.087245, 40 to 86.654484, 50 fc rgb "#808080"
set object 57 rect from 0.091459, 40 to 87.222841, 50 fc rgb "#800080"
set object 58 rect from 0.092271, 40 to 87.850030, 50 fc rgb "#008080"
set object 59 rect from 0.092703, 40 to 149.156046, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.159536, 50 to 159.989991, 60 fc rgb "#FF0000"
set object 61 rect from 0.100830, 50 to 96.862188, 60 fc rgb "#00FF00"
set object 62 rect from 0.102409, 50 to 97.407781, 60 fc rgb "#0000FF"
set object 63 rect from 0.102767, 50 to 100.196431, 60 fc rgb "#FFFF00"
set object 64 rect from 0.105708, 50 to 101.639635, 60 fc rgb "#FF00FF"
set object 65 rect from 0.107226, 50 to 105.563119, 60 fc rgb "#808080"
set object 66 rect from 0.111363, 50 to 106.094474, 60 fc rgb "#800080"
set object 67 rect from 0.112164, 50 to 106.670424, 60 fc rgb "#008080"
set object 68 rect from 0.112555, 50 to 150.814628, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
