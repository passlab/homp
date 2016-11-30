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

set object 15 rect from 0.523118, 0 to 159.717432, 10 fc rgb "#FF0000"
set object 16 rect from 0.400328, 0 to 113.170605, 10 fc rgb "#00FF00"
set object 17 rect from 0.402982, 0 to 113.382824, 10 fc rgb "#0000FF"
set object 18 rect from 0.403496, 0 to 113.603185, 10 fc rgb "#FFFF00"
set object 19 rect from 0.404323, 0 to 113.935988, 10 fc rgb "#FF00FF"
set object 20 rect from 0.405499, 0 to 125.893689, 10 fc rgb "#808080"
set object 21 rect from 0.448068, 0 to 126.083131, 10 fc rgb "#800080"
set object 22 rect from 0.448965, 0 to 126.257117, 10 fc rgb "#008080"
set object 23 rect from 0.449312, 0 to 146.868420, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.524777, 10 to 159.939745, 20 fc rgb "#FF0000"
set object 25 rect from 0.459446, 10 to 129.678425, 20 fc rgb "#00FF00"
set object 26 rect from 0.461729, 10 to 129.897102, 20 fc rgb "#0000FF"
set object 27 rect from 0.462249, 10 to 130.103415, 20 fc rgb "#FFFF00"
set object 28 rect from 0.462997, 10 to 130.392650, 20 fc rgb "#FF00FF"
set object 29 rect from 0.464022, 10 to 142.169612, 20 fc rgb "#808080"
set object 30 rect from 0.505938, 10 to 142.327859, 20 fc rgb "#800080"
set object 31 rect from 0.506747, 10 to 142.478516, 20 fc rgb "#008080"
set object 32 rect from 0.507020, 10 to 147.337263, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.517244, 20 to 168.280237, 30 fc rgb "#FF0000"
set object 34 rect from 0.038493, 20 to 12.010542, 30 fc rgb "#00FF00"
set object 35 rect from 0.043385, 20 to 12.455770, 30 fc rgb "#0000FF"
set object 36 rect from 0.044451, 20 to 14.012956, 30 fc rgb "#FFFF00"
set object 37 rect from 0.050026, 20 to 22.847858, 30 fc rgb "#FF00FF"
set object 38 rect from 0.081412, 20 to 34.691432, 30 fc rgb "#808080"
set object 39 rect from 0.123572, 20 to 35.219583, 30 fc rgb "#800080"
set object 40 rect from 0.125880, 20 to 35.558568, 30 fc rgb "#008080"
set object 41 rect from 0.126647, 20 to 145.156363, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.521472, 30 to 166.471219, 40 fc rgb "#FF0000"
set object 43 rect from 0.313708, 30 to 88.675628, 40 fc rgb "#00FF00"
set object 44 rect from 0.315868, 30 to 88.881372, 40 fc rgb "#0000FF"
set object 45 rect from 0.316331, 30 to 89.786170, 40 fc rgb "#FFFF00"
set object 46 rect from 0.319567, 30 to 96.584932, 40 fc rgb "#FF00FF"
set object 47 rect from 0.343753, 30 to 108.415020, 40 fc rgb "#808080"
set object 48 rect from 0.385842, 30 to 108.793636, 40 fc rgb "#800080"
set object 49 rect from 0.387489, 30 to 108.989260, 40 fc rgb "#008080"
set object 50 rect from 0.387890, 30 to 146.375125, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.514867, 40 to 164.943829, 50 fc rgb "#FF0000"
set object 52 rect from 0.138850, 40 to 39.628318, 50 fc rgb "#00FF00"
set object 53 rect from 0.141361, 40 to 39.876800, 50 fc rgb "#0000FF"
set object 54 rect from 0.142007, 40 to 40.790871, 50 fc rgb "#FFFF00"
set object 55 rect from 0.145256, 40 to 47.790602, 50 fc rgb "#FF00FF"
set object 56 rect from 0.170155, 40 to 59.673530, 50 fc rgb "#808080"
set object 57 rect from 0.212429, 40 to 60.067325, 50 fc rgb "#800080"
set object 58 rect from 0.214155, 40 to 60.271385, 50 fc rgb "#008080"
set object 59 rect from 0.214560, 40 to 144.370750, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.519529, 50 to 167.105328, 60 fc rgb "#FF0000"
set object 61 rect from 0.225273, 50 to 63.879328, 60 fc rgb "#00FF00"
set object 62 rect from 0.227635, 50 to 64.059220, 60 fc rgb "#0000FF"
set object 63 rect from 0.228038, 50 to 64.956421, 60 fc rgb "#FFFF00"
set object 64 rect from 0.231219, 50 to 72.931499, 60 fc rgb "#FF00FF"
set object 65 rect from 0.259586, 50 to 84.781826, 60 fc rgb "#808080"
set object 66 rect from 0.301782, 50 to 85.175059, 60 fc rgb "#800080"
set object 67 rect from 0.303438, 50 to 85.364225, 60 fc rgb "#008080"
set object 68 rect from 0.303822, 50 to 145.868059, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
