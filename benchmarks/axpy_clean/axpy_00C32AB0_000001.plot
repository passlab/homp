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

set object 15 rect from 0.105352, 0 to 156.160881, 10 fc rgb "#FF0000"
set object 16 rect from 0.067505, 0 to 100.137271, 10 fc rgb "#00FF00"
set object 17 rect from 0.069352, 0 to 100.961149, 10 fc rgb "#0000FF"
set object 18 rect from 0.069705, 0 to 101.859000, 10 fc rgb "#FFFF00"
set object 19 rect from 0.070334, 0 to 103.206502, 10 fc rgb "#FF00FF"
set object 20 rect from 0.071263, 0 to 103.747534, 10 fc rgb "#808080"
set object 21 rect from 0.071636, 0 to 104.422011, 10 fc rgb "#800080"
set object 22 rect from 0.072310, 0 to 105.108091, 10 fc rgb "#008080"
set object 23 rect from 0.072566, 0 to 152.086464, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.101032, 10 to 152.469390, 20 fc rgb "#FF0000"
set object 25 rect from 0.022725, 10 to 36.629148, 20 fc rgb "#00FF00"
set object 26 rect from 0.025632, 10 to 38.068031, 20 fc rgb "#0000FF"
set object 27 rect from 0.026357, 10 to 40.007334, 20 fc rgb "#FFFF00"
set object 28 rect from 0.027731, 10 to 42.049619, 20 fc rgb "#FF00FF"
set object 29 rect from 0.029106, 10 to 42.802422, 20 fc rgb "#808080"
set object 30 rect from 0.029634, 10 to 43.679967, 20 fc rgb "#800080"
set object 31 rect from 0.030583, 10 to 44.696757, 20 fc rgb "#008080"
set object 32 rect from 0.030932, 10 to 145.327196, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.108032, 20 to 159.903143, 30 fc rgb "#FF0000"
set object 34 rect from 0.090711, 20 to 133.382435, 30 fc rgb "#00FF00"
set object 35 rect from 0.092256, 20 to 134.113481, 30 fc rgb "#0000FF"
set object 36 rect from 0.092562, 20 to 134.991026, 30 fc rgb "#FFFF00"
set object 37 rect from 0.093174, 20 to 136.295014, 30 fc rgb "#FF00FF"
set object 38 rect from 0.094081, 20 to 136.844748, 30 fc rgb "#808080"
set object 39 rect from 0.094455, 20 to 137.507621, 30 fc rgb "#800080"
set object 40 rect from 0.095139, 20 to 138.272027, 30 fc rgb "#008080"
set object 41 rect from 0.095429, 20 to 156.107214, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.106820, 30 to 158.483110, 40 fc rgb "#FF0000"
set object 43 rect from 0.078687, 30 to 116.073773, 40 fc rgb "#00FF00"
set object 44 rect from 0.080326, 30 to 116.946965, 40 fc rgb "#0000FF"
set object 45 rect from 0.080747, 30 to 118.015974, 40 fc rgb "#FFFF00"
set object 46 rect from 0.081479, 30 to 119.418595, 40 fc rgb "#FF00FF"
set object 47 rect from 0.082445, 30 to 119.901610, 40 fc rgb "#808080"
set object 48 rect from 0.082792, 30 to 120.606546, 40 fc rgb "#800080"
set object 49 rect from 0.083467, 30 to 121.331788, 40 fc rgb "#008080"
set object 50 rect from 0.083752, 30 to 154.227384, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.104014, 40 to 154.447857, 50 fc rgb "#FF0000"
set object 52 rect from 0.054354, 40 to 80.800829, 50 fc rgb "#00FF00"
set object 53 rect from 0.056008, 40 to 81.576840, 50 fc rgb "#0000FF"
set object 54 rect from 0.056341, 40 to 82.515306, 50 fc rgb "#FFFF00"
set object 55 rect from 0.056993, 40 to 84.016559, 50 fc rgb "#FF00FF"
set object 56 rect from 0.058044, 40 to 84.573548, 50 fc rgb "#808080"
set object 57 rect from 0.058420, 40 to 85.307493, 50 fc rgb "#800080"
set object 58 rect from 0.059141, 40 to 86.053044, 50 fc rgb "#008080"
set object 59 rect from 0.059436, 40 to 150.057234, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.102580, 50 to 153.148219, 60 fc rgb "#FF0000"
set object 61 rect from 0.040542, 50 to 61.174290, 60 fc rgb "#00FF00"
set object 62 rect from 0.042481, 50 to 62.064890, 60 fc rgb "#0000FF"
set object 63 rect from 0.042890, 50 to 63.316659, 60 fc rgb "#FFFF00"
set object 64 rect from 0.043775, 50 to 65.086254, 60 fc rgb "#FF00FF"
set object 65 rect from 0.044980, 50 to 65.631636, 60 fc rgb "#808080"
set object 66 rect from 0.045362, 50 to 66.477271, 60 fc rgb "#800080"
set object 67 rect from 0.046173, 50 to 67.660869, 60 fc rgb "#008080"
set object 68 rect from 0.046771, 50 to 148.017848, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
