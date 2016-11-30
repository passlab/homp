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

set object 15 rect from 0.130070, 0 to 160.968927, 10 fc rgb "#FF0000"
set object 16 rect from 0.108043, 0 to 133.271164, 10 fc rgb "#00FF00"
set object 17 rect from 0.110260, 0 to 134.095899, 10 fc rgb "#0000FF"
set object 18 rect from 0.110675, 0 to 134.961871, 10 fc rgb "#FFFF00"
set object 19 rect from 0.111393, 0 to 136.223230, 10 fc rgb "#FF00FF"
set object 20 rect from 0.112436, 0 to 136.715646, 10 fc rgb "#808080"
set object 21 rect from 0.112843, 0 to 137.370582, 10 fc rgb "#800080"
set object 22 rect from 0.113652, 0 to 138.041287, 10 fc rgb "#008080"
set object 23 rect from 0.113953, 0 to 157.081756, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.128385, 10 to 159.186044, 20 fc rgb "#FF0000"
set object 25 rect from 0.093170, 10 to 115.361066, 20 fc rgb "#00FF00"
set object 26 rect from 0.095464, 10 to 116.137286, 20 fc rgb "#0000FF"
set object 27 rect from 0.095868, 10 to 117.008111, 20 fc rgb "#FFFF00"
set object 28 rect from 0.096625, 10 to 118.474440, 20 fc rgb "#FF00FF"
set object 29 rect from 0.097828, 10 to 119.060246, 20 fc rgb "#808080"
set object 30 rect from 0.098299, 10 to 119.775825, 20 fc rgb "#800080"
set object 31 rect from 0.099141, 10 to 120.530216, 20 fc rgb "#008080"
set object 32 rect from 0.099497, 10 to 154.897421, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.126668, 20 to 156.928937, 30 fc rgb "#FF0000"
set object 34 rect from 0.079458, 20 to 98.477036, 30 fc rgb "#00FF00"
set object 35 rect from 0.081577, 20 to 99.204742, 30 fc rgb "#0000FF"
set object 36 rect from 0.081937, 20 to 100.269622, 30 fc rgb "#FFFF00"
set object 37 rect from 0.082787, 20 to 101.573431, 30 fc rgb "#FF00FF"
set object 38 rect from 0.083861, 20 to 102.046442, 30 fc rgb "#808080"
set object 39 rect from 0.084276, 20 to 102.717146, 30 fc rgb "#800080"
set object 40 rect from 0.085091, 20 to 103.592819, 30 fc rgb "#008080"
set object 41 rect from 0.085531, 20 to 153.055108, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.125043, 30 to 155.428649, 40 fc rgb "#FF0000"
set object 43 rect from 0.064117, 30 to 80.029646, 40 fc rgb "#00FF00"
set object 44 rect from 0.066349, 30 to 80.781611, 40 fc rgb "#0000FF"
set object 45 rect from 0.066718, 30 to 81.921687, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067676, 30 to 83.331014, 40 fc rgb "#FF00FF"
set object 47 rect from 0.068820, 30 to 83.848900, 40 fc rgb "#808080"
set object 48 rect from 0.069271, 30 to 84.677274, 40 fc rgb "#800080"
set object 49 rect from 0.070220, 30 to 85.471687, 40 fc rgb "#008080"
set object 50 rect from 0.070602, 30 to 150.962949, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.123387, 40 to 153.309808, 50 fc rgb "#FF0000"
set object 52 rect from 0.049126, 40 to 62.252962, 50 fc rgb "#00FF00"
set object 53 rect from 0.051711, 40 to 63.165022, 50 fc rgb "#0000FF"
set object 54 rect from 0.052191, 40 to 64.181388, 50 fc rgb "#FFFF00"
set object 55 rect from 0.053055, 40 to 65.614973, 50 fc rgb "#FF00FF"
set object 56 rect from 0.054251, 40 to 66.186223, 50 fc rgb "#808080"
set object 57 rect from 0.054693, 40 to 66.871481, 50 fc rgb "#800080"
set object 58 rect from 0.055551, 40 to 67.847822, 50 fc rgb "#008080"
set object 59 rect from 0.056108, 40 to 148.858662, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.121447, 50 to 154.335872, 60 fc rgb "#FF0000"
set object 61 rect from 0.026300, 50 to 35.800791, 60 fc rgb "#00FF00"
set object 62 rect from 0.029935, 50 to 37.181009, 60 fc rgb "#0000FF"
set object 63 rect from 0.030787, 50 to 39.966916, 60 fc rgb "#FFFF00"
set object 64 rect from 0.033121, 50 to 42.063927, 60 fc rgb "#FF00FF"
set object 65 rect from 0.034816, 50 to 42.906854, 60 fc rgb "#808080"
set object 66 rect from 0.035597, 50 to 43.957181, 60 fc rgb "#800080"
set object 67 rect from 0.037002, 50 to 45.522964, 60 fc rgb "#008080"
set object 68 rect from 0.037886, 50 to 145.444499, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
