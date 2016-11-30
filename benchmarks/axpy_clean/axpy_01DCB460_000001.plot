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

set object 15 rect from 0.264250, 0 to 154.753947, 10 fc rgb "#FF0000"
set object 16 rect from 0.078561, 0 to 44.513860, 10 fc rgb "#00FF00"
set object 17 rect from 0.080771, 0 to 44.967109, 10 fc rgb "#0000FF"
set object 18 rect from 0.081359, 0 to 45.329040, 10 fc rgb "#FFFF00"
set object 19 rect from 0.082056, 0 to 45.950526, 10 fc rgb "#FF00FF"
set object 20 rect from 0.083159, 0 to 53.118897, 10 fc rgb "#808080"
set object 21 rect from 0.096114, 0 to 53.436557, 10 fc rgb "#800080"
set object 22 rect from 0.096942, 0 to 53.759750, 10 fc rgb "#008080"
set object 23 rect from 0.097271, 0 to 145.798597, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.267747, 10 to 156.644393, 20 fc rgb "#FF0000"
set object 25 rect from 0.157184, 10 to 87.785861, 20 fc rgb "#00FF00"
set object 26 rect from 0.158960, 10 to 88.147792, 20 fc rgb "#0000FF"
set object 27 rect from 0.159397, 10 to 88.521899, 20 fc rgb "#FFFF00"
set object 28 rect from 0.160082, 10 to 89.063139, 20 fc rgb "#FF00FF"
set object 29 rect from 0.161055, 10 to 96.345514, 20 fc rgb "#808080"
set object 30 rect from 0.174215, 10 to 96.648231, 20 fc rgb "#800080"
set object 31 rect from 0.175030, 10 to 96.943757, 20 fc rgb "#008080"
set object 32 rect from 0.175284, 10 to 147.843460, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.266066, 20 to 169.230146, 30 fc rgb "#FF0000"
set object 34 rect from 0.105379, 20 to 59.180995, 30 fc rgb "#00FF00"
set object 35 rect from 0.107280, 20 to 59.526326, 30 fc rgb "#0000FF"
set object 36 rect from 0.107667, 20 to 61.342628, 30 fc rgb "#FFFF00"
set object 37 rect from 0.110976, 20 to 73.767326, 30 fc rgb "#FF00FF"
set object 38 rect from 0.133408, 20 to 81.031996, 30 fc rgb "#808080"
set object 39 rect from 0.146570, 20 to 81.566041, 30 fc rgb "#800080"
set object 40 rect from 0.147759, 20 to 81.960624, 30 fc rgb "#008080"
set object 41 rect from 0.148210, 20 to 146.925900, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.262269, 30 to 160.648908, 40 fc rgb "#FF0000"
set object 43 rect from 0.036059, 30 to 21.688298, 40 fc rgb "#00FF00"
set object 44 rect from 0.039605, 30 to 22.356269, 40 fc rgb "#0000FF"
set object 45 rect from 0.040518, 30 to 24.656811, 40 fc rgb "#FFFF00"
set object 46 rect from 0.044688, 30 to 29.918672, 40 fc rgb "#FF00FF"
set object 47 rect from 0.054181, 30 to 37.114162, 40 fc rgb "#808080"
set object 48 rect from 0.067198, 30 to 37.637139, 40 fc rgb "#800080"
set object 49 rect from 0.068519, 30 to 38.247554, 40 fc rgb "#008080"
set object 50 rect from 0.069232, 30 to 144.605434, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.270592, 40 to 167.258879, 50 fc rgb "#FF0000"
set object 52 rect from 0.220726, 40 to 122.835232, 50 fc rgb "#00FF00"
set object 53 rect from 0.222289, 40 to 123.175582, 50 fc rgb "#0000FF"
set object 54 rect from 0.222682, 40 to 124.960342, 50 fc rgb "#FFFF00"
set object 55 rect from 0.225936, 40 to 132.877484, 50 fc rgb "#FF00FF"
set object 56 rect from 0.240223, 40 to 140.284930, 50 fc rgb "#808080"
set object 57 rect from 0.253611, 40 to 140.734855, 50 fc rgb "#800080"
set object 58 rect from 0.254705, 40 to 141.104538, 50 fc rgb "#008080"
set object 59 rect from 0.255089, 40 to 149.509235, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.269250, 50 to 163.467995, 60 fc rgb "#FF0000"
set object 61 rect from 0.183470, 50 to 102.268150, 60 fc rgb "#00FF00"
set object 62 rect from 0.185123, 50 to 102.724719, 60 fc rgb "#0000FF"
set object 63 rect from 0.185728, 50 to 104.477383, 60 fc rgb "#FFFF00"
set object 64 rect from 0.188929, 50 to 109.333036, 60 fc rgb "#FF00FF"
set object 65 rect from 0.197670, 50 to 116.645849, 60 fc rgb "#808080"
set object 66 rect from 0.210885, 50 to 117.110165, 60 fc rgb "#800080"
set object 67 rect from 0.211977, 50 to 117.450515, 60 fc rgb "#008080"
set object 68 rect from 0.212341, 50 to 148.695164, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
