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

set object 15 rect from 1.431914, 0 to 272.315467, 10 fc rgb "#FF0000"
set object 16 rect from 0.159986, 0 to 16.432359, 10 fc rgb "#00FF00"
set object 17 rect from 0.161811, 0 to 16.495263, 10 fc rgb "#0000FF"
set object 18 rect from 0.162176, 0 to 16.561527, 10 fc rgb "#FFFF00"
set object 19 rect from 0.162832, 0 to 141.971679, 10 fc rgb "#FF00FF"
set object 20 rect from 1.394922, 0 to 143.009812, 10 fc rgb "#808080"
set object 21 rect from 1.405223, 0 to 143.085847, 10 fc rgb "#800080"
set object 22 rect from 1.406337, 0 to 143.180102, 10 fc rgb "#008080"
set object 23 rect from 1.406812, 0 to 145.663817, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.429860, 10 to 146.697572, 20 fc rgb "#FF0000"
set object 25 rect from 0.138095, 10 to 14.223471, 20 fc rgb "#00FF00"
set object 26 rect from 0.140105, 10 to 14.290753, 20 fc rgb "#0000FF"
set object 27 rect from 0.140518, 10 to 14.364650, 20 fc rgb "#FFFF00"
set object 28 rect from 0.141285, 10 to 14.489136, 20 fc rgb "#FF00FF"
set object 29 rect from 0.142496, 10 to 15.402678, 20 fc rgb "#808080"
set object 30 rect from 0.151459, 10 to 15.462224, 20 fc rgb "#800080"
set object 31 rect from 0.152306, 10 to 15.523703, 20 fc rgb "#008080"
set object 32 rect from 0.152650, 10 to 145.448638, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.426920, 20 to 147.030110, 30 fc rgb "#FF0000"
set object 34 rect from 0.110942, 20 to 11.433171, 30 fc rgb "#00FF00"
set object 35 rect from 0.112690, 20 to 11.499333, 30 fc rgb "#0000FF"
set object 36 rect from 0.113094, 20 to 11.819659, 30 fc rgb "#FFFF00"
set object 37 rect from 0.116246, 20 to 12.317807, 30 fc rgb "#FF00FF"
set object 38 rect from 0.121137, 20 to 13.226462, 30 fc rgb "#808080"
set object 39 rect from 0.130081, 20 to 13.296187, 30 fc rgb "#800080"
set object 40 rect from 0.131016, 20 to 13.368557, 30 fc rgb "#008080"
set object 41 rect from 0.131463, 20 to 145.162615, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.424092, 30 to 146.764445, 40 fc rgb "#FF0000"
set object 43 rect from 0.082105, 30 to 8.514214, 40 fc rgb "#00FF00"
set object 44 rect from 0.084017, 30 to 8.576304, 40 fc rgb "#0000FF"
set object 45 rect from 0.084406, 30 to 8.914034, 40 fc rgb "#FFFF00"
set object 46 rect from 0.087742, 30 to 9.420326, 40 fc rgb "#FF00FF"
set object 47 rect from 0.092669, 30 to 10.335599, 40 fc rgb "#808080"
set object 48 rect from 0.101670, 30 to 10.406849, 40 fc rgb "#800080"
set object 49 rect from 0.102659, 30 to 10.486244, 40 fc rgb "#008080"
set object 50 rect from 0.103143, 30 to 144.867431, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.421477, 40 to 146.527379, 50 fc rgb "#FF0000"
set object 52 rect from 0.053163, 40 to 5.604722, 50 fc rgb "#00FF00"
set object 53 rect from 0.055427, 40 to 5.673225, 50 fc rgb "#0000FF"
set object 54 rect from 0.055857, 40 to 5.999962, 50 fc rgb "#FFFF00"
set object 55 rect from 0.059103, 40 to 6.534550, 50 fc rgb "#FF00FF"
set object 56 rect from 0.064333, 40 to 7.448906, 50 fc rgb "#808080"
set object 57 rect from 0.073352, 40 to 7.527588, 50 fc rgb "#800080"
set object 58 rect from 0.074352, 40 to 7.613904, 50 fc rgb "#008080"
set object 59 rect from 0.074941, 40 to 144.591181, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.418474, 50 to 146.447784, 60 fc rgb "#FF0000"
set object 61 rect from 0.019573, 50 to 2.262531, 60 fc rgb "#00FF00"
set object 62 rect from 0.022716, 50 to 2.372970, 60 fc rgb "#0000FF"
set object 63 rect from 0.023445, 50 to 2.824296, 60 fc rgb "#FFFF00"
set object 64 rect from 0.027907, 50 to 3.405910, 60 fc rgb "#FF00FF"
set object 65 rect from 0.033604, 50 to 4.339911, 60 fc rgb "#808080"
set object 66 rect from 0.042786, 50 to 4.423173, 60 fc rgb "#800080"
set object 67 rect from 0.044033, 50 to 4.551120, 60 fc rgb "#008080"
set object 68 rect from 0.044852, 50 to 144.260270, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
