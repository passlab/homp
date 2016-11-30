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

set object 15 rect from 0.390985, 0 to 157.320149, 10 fc rgb "#FF0000"
set object 16 rect from 0.151582, 0 to 57.216932, 10 fc rgb "#00FF00"
set object 17 rect from 0.153486, 0 to 57.445934, 10 fc rgb "#0000FF"
set object 18 rect from 0.153875, 0 to 57.667466, 10 fc rgb "#FFFF00"
set object 19 rect from 0.154478, 0 to 58.010407, 10 fc rgb "#FF00FF"
set object 20 rect from 0.155403, 0 to 68.541826, 10 fc rgb "#808080"
set object 21 rect from 0.183605, 0 to 68.730481, 10 fc rgb "#800080"
set object 22 rect from 0.184364, 0 to 68.928844, 10 fc rgb "#008080"
set object 23 rect from 0.184626, 0 to 145.837228, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.387423, 10 to 156.781461, 20 fc rgb "#FF0000"
set object 25 rect from 0.035320, 10 to 14.594408, 20 fc rgb "#00FF00"
set object 26 rect from 0.039470, 10 to 15.100230, 20 fc rgb "#0000FF"
set object 27 rect from 0.040532, 10 to 15.706909, 20 fc rgb "#FFFF00"
set object 28 rect from 0.042170, 10 to 16.232527, 20 fc rgb "#FF00FF"
set object 29 rect from 0.043559, 10 to 26.707911, 20 fc rgb "#808080"
set object 30 rect from 0.071625, 10 to 26.933919, 20 fc rgb "#800080"
set object 31 rect from 0.072595, 10 to 27.187203, 20 fc rgb "#008080"
set object 32 rect from 0.072890, 10 to 144.403451, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.395286, 20 to 168.362612, 30 fc rgb "#FF0000"
set object 34 rect from 0.322580, 20 to 120.980005, 30 fc rgb "#00FF00"
set object 35 rect from 0.324148, 20 to 121.190704, 30 fc rgb "#0000FF"
set object 36 rect from 0.324510, 20 to 122.119784, 30 fc rgb "#FFFF00"
set object 37 rect from 0.327001, 20 to 131.144947, 30 fc rgb "#FF00FF"
set object 38 rect from 0.351155, 20 to 141.381243, 30 fc rgb "#808080"
set object 39 rect from 0.378582, 20 to 141.911347, 30 fc rgb "#800080"
set object 40 rect from 0.380246, 20 to 142.164253, 30 fc rgb "#008080"
set object 41 rect from 0.380653, 20 to 147.520181, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.394048, 30 to 166.470841, 40 fc rgb "#FF0000"
set object 43 rect from 0.260376, 30 to 97.731765, 40 fc rgb "#00FF00"
set object 44 rect from 0.261915, 30 to 97.983179, 40 fc rgb "#0000FF"
set object 45 rect from 0.262386, 30 to 98.939157, 40 fc rgb "#FFFF00"
set object 46 rect from 0.264946, 30 to 106.587702, 40 fc rgb "#FF00FF"
set object 47 rect from 0.285426, 30 to 116.714913, 40 fc rgb "#808080"
set object 48 rect from 0.312543, 30 to 117.230443, 40 fc rgb "#800080"
set object 49 rect from 0.314148, 30 to 117.444126, 40 fc rgb "#008080"
set object 50 rect from 0.314482, 30 to 147.032295, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.389359, 40 to 167.275906, 50 fc rgb "#FF0000"
set object 52 rect from 0.081821, 40 to 31.209844, 50 fc rgb "#00FF00"
set object 53 rect from 0.083850, 40 to 31.477702, 50 fc rgb "#0000FF"
set object 54 rect from 0.084360, 40 to 32.675752, 50 fc rgb "#FFFF00"
set object 55 rect from 0.087595, 40 to 42.550057, 50 fc rgb "#FF00FF"
set object 56 rect from 0.114076, 40 to 52.762817, 50 fc rgb "#808080"
set object 57 rect from 0.141372, 40 to 53.308608, 50 fc rgb "#800080"
set object 58 rect from 0.143078, 40 to 53.688156, 50 fc rgb "#008080"
set object 59 rect from 0.143838, 40 to 145.223446, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.392610, 50 to 166.246693, 60 fc rgb "#FF0000"
set object 61 rect from 0.192768, 50 to 72.546910, 60 fc rgb "#00FF00"
set object 62 rect from 0.194505, 50 to 72.745273, 60 fc rgb "#0000FF"
set object 63 rect from 0.194828, 50 to 73.729273, 60 fc rgb "#FFFF00"
set object 64 rect from 0.197475, 50 to 81.680412, 60 fc rgb "#FF00FF"
set object 65 rect from 0.218767, 50 to 91.861052, 60 fc rgb "#808080"
set object 66 rect from 0.246035, 50 to 92.378075, 60 fc rgb "#800080"
set object 67 rect from 0.247650, 50 to 92.631726, 60 fc rgb "#008080"
set object 68 rect from 0.248064, 50 to 146.424858, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
