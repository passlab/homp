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

set object 15 rect from 0.216764, 0 to 152.523230, 10 fc rgb "#FF0000"
set object 16 rect from 0.088506, 0 to 60.987137, 10 fc rgb "#00FF00"
set object 17 rect from 0.091191, 0 to 61.581938, 10 fc rgb "#0000FF"
set object 18 rect from 0.091844, 0 to 62.184122, 10 fc rgb "#FFFF00"
set object 19 rect from 0.092785, 0 to 63.030671, 10 fc rgb "#FF00FF"
set object 20 rect from 0.094041, 0 to 68.118695, 10 fc rgb "#808080"
set object 21 rect from 0.101627, 0 to 68.520152, 10 fc rgb "#800080"
set object 22 rect from 0.102469, 0 to 68.935706, 10 fc rgb "#008080"
set object 23 rect from 0.102924, 0 to 144.733094, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.222571, 10 to 156.094715, 20 fc rgb "#FF0000"
set object 25 rect from 0.141263, 10 to 96.118602, 20 fc rgb "#00FF00"
set object 26 rect from 0.143554, 10 to 96.567722, 20 fc rgb "#0000FF"
set object 27 rect from 0.143956, 10 to 97.126942, 20 fc rgb "#FFFF00"
set object 28 rect from 0.144791, 10 to 97.830498, 20 fc rgb "#FF00FF"
set object 29 rect from 0.145858, 10 to 102.894354, 20 fc rgb "#808080"
set object 30 rect from 0.153401, 10 to 103.301181, 20 fc rgb "#800080"
set object 31 rect from 0.154245, 10 to 103.662357, 20 fc rgb "#008080"
set object 32 rect from 0.154529, 10 to 148.922876, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.224278, 20 to 160.728245, 30 fc rgb "#FF0000"
set object 34 rect from 0.162809, 20 to 110.456235, 30 fc rgb "#00FF00"
set object 35 rect from 0.164885, 20 to 110.890586, 30 fc rgb "#0000FF"
set object 36 rect from 0.165292, 20 to 113.092555, 30 fc rgb "#FFFF00"
set object 37 rect from 0.168575, 20 to 115.566412, 30 fc rgb "#FF00FF"
set object 38 rect from 0.172261, 20 to 120.644367, 30 fc rgb "#808080"
set object 39 rect from 0.179835, 20 to 121.094158, 30 fc rgb "#800080"
set object 40 rect from 0.180753, 20 to 121.568119, 30 fc rgb "#008080"
set object 41 rect from 0.181205, 20 to 150.151413, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.225834, 30 to 161.317005, 40 fc rgb "#FF0000"
set object 43 rect from 0.191006, 30 to 129.395848, 40 fc rgb "#00FF00"
set object 44 rect from 0.193096, 30 to 129.823486, 40 fc rgb "#0000FF"
set object 45 rect from 0.193495, 30 to 132.008000, 40 fc rgb "#FFFF00"
set object 46 rect from 0.196762, 30 to 134.146192, 40 fc rgb "#FF00FF"
set object 47 rect from 0.199936, 30 to 139.145599, 40 fc rgb "#808080"
set object 48 rect from 0.207394, 30 to 139.585322, 40 fc rgb "#800080"
set object 49 rect from 0.208289, 30 to 139.986779, 40 fc rgb "#008080"
set object 50 rect from 0.208636, 30 to 151.254411, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.218825, 40 to 160.047514, 50 fc rgb "#FF0000"
set object 52 rect from 0.051514, 40 to 37.608678, 50 fc rgb "#00FF00"
set object 53 rect from 0.056551, 40 to 38.677439, 50 fc rgb "#0000FF"
set object 54 rect from 0.057744, 40 to 42.305987, 50 fc rgb "#FFFF00"
set object 55 rect from 0.063182, 40 to 45.446477, 50 fc rgb "#FF00FF"
set object 56 rect from 0.067825, 40 to 50.835929, 50 fc rgb "#808080"
set object 57 rect from 0.075870, 40 to 51.416631, 50 fc rgb "#800080"
set object 58 rect from 0.077405, 40 to 52.384020, 50 fc rgb "#008080"
set object 59 rect from 0.078160, 40 to 146.365102, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.220656, 50 to 158.140260, 60 fc rgb "#FF0000"
set object 61 rect from 0.112926, 50 to 77.174961, 60 fc rgb "#00FF00"
set object 62 rect from 0.115436, 50 to 77.746935, 60 fc rgb "#0000FF"
set object 63 rect from 0.115926, 50 to 79.853575, 60 fc rgb "#FFFF00"
set object 64 rect from 0.119081, 50 to 82.250900, 60 fc rgb "#FF00FF"
set object 65 rect from 0.122659, 50 to 87.306700, 60 fc rgb "#808080"
set object 66 rect from 0.130205, 50 to 87.792074, 60 fc rgb "#800080"
set object 67 rect from 0.131157, 50 to 88.278117, 60 fc rgb "#008080"
set object 68 rect from 0.131645, 50 to 147.694338, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
