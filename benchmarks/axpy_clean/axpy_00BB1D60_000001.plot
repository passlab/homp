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

set object 15 rect from 0.272105, 0 to 155.022307, 10 fc rgb "#FF0000"
set object 16 rect from 0.182317, 0 to 100.565260, 10 fc rgb "#00FF00"
set object 17 rect from 0.184517, 0 to 101.104443, 10 fc rgb "#0000FF"
set object 18 rect from 0.185201, 0 to 101.480285, 10 fc rgb "#FFFF00"
set object 19 rect from 0.185908, 0 to 102.070263, 10 fc rgb "#FF00FF"
set object 20 rect from 0.186983, 0 to 107.100415, 10 fc rgb "#808080"
set object 21 rect from 0.196174, 0 to 107.392128, 10 fc rgb "#800080"
set object 22 rect from 0.196976, 0 to 107.681659, 10 fc rgb "#008080"
set object 23 rect from 0.197262, 0 to 148.357137, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.270536, 10 to 154.763896, 20 fc rgb "#FF0000"
set object 25 rect from 0.105026, 10 to 83.904262, 20 fc rgb "#00FF00"
set object 26 rect from 0.154446, 10 to 84.631362, 20 fc rgb "#0000FF"
set object 27 rect from 0.155056, 10 to 85.148689, 20 fc rgb "#FFFF00"
set object 28 rect from 0.156041, 10 to 86.092114, 20 fc rgb "#FF00FF"
set object 29 rect from 0.157735, 10 to 91.241350, 20 fc rgb "#808080"
set object 30 rect from 0.167164, 10 to 91.597525, 20 fc rgb "#800080"
set object 31 rect from 0.168169, 10 to 92.001231, 20 fc rgb "#008080"
set object 32 rect from 0.168573, 10 to 147.348149, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.266119, 20 to 157.755879, 30 fc rgb "#FF0000"
set object 34 rect from 0.034600, 20 to 20.586019, 30 fc rgb "#00FF00"
set object 35 rect from 0.038265, 20 to 21.470990, 30 fc rgb "#0000FF"
set object 36 rect from 0.039444, 20 to 24.128101, 30 fc rgb "#FFFF00"
set object 37 rect from 0.044339, 20 to 27.984835, 30 fc rgb "#FF00FF"
set object 38 rect from 0.051368, 20 to 33.040662, 30 fc rgb "#808080"
set object 39 rect from 0.060731, 20 to 33.591860, 30 fc rgb "#800080"
set object 40 rect from 0.062075, 20 to 34.292740, 30 fc rgb "#008080"
set object 41 rect from 0.062920, 20 to 144.785009, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.275169, 30 to 160.384581, 40 fc rgb "#FF0000"
set object 43 rect from 0.236646, 30 to 130.117931, 40 fc rgb "#00FF00"
set object 44 rect from 0.238574, 30 to 130.511250, 40 fc rgb "#0000FF"
set object 45 rect from 0.239029, 30 to 132.214556, 40 fc rgb "#FFFF00"
set object 46 rect from 0.242148, 30 to 135.354023, 40 fc rgb "#FF00FF"
set object 47 rect from 0.247908, 30 to 140.241047, 40 fc rgb "#808080"
set object 48 rect from 0.256867, 30 to 140.613062, 40 fc rgb "#800080"
set object 49 rect from 0.257840, 30 to 141.010207, 40 fc rgb "#008080"
set object 50 rect from 0.258252, 30 to 150.056063, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.273682, 40 to 160.039346, 50 fc rgb "#FF0000"
set object 52 rect from 0.205162, 40 to 112.965281, 50 fc rgb "#00FF00"
set object 53 rect from 0.207156, 40 to 113.388646, 50 fc rgb "#0000FF"
set object 54 rect from 0.207685, 40 to 115.256921, 50 fc rgb "#FFFF00"
set object 55 rect from 0.211153, 40 to 118.636761, 50 fc rgb "#FF00FF"
set object 56 rect from 0.217293, 40 to 123.537989, 50 fc rgb "#808080"
set object 57 rect from 0.226289, 40 to 123.934588, 50 fc rgb "#800080"
set object 58 rect from 0.227278, 40 to 124.409846, 50 fc rgb "#008080"
set object 59 rect from 0.227866, 40 to 149.188569, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.268653, 50 to 157.144045, 60 fc rgb "#FF0000"
set object 61 rect from 0.073456, 50 to 41.153471, 60 fc rgb "#00FF00"
set object 62 rect from 0.075709, 50 to 41.520570, 60 fc rgb "#0000FF"
set object 63 rect from 0.076127, 50 to 43.308542, 60 fc rgb "#FFFF00"
set object 64 rect from 0.079432, 50 to 46.633207, 60 fc rgb "#FF00FF"
set object 65 rect from 0.085500, 50 to 51.578132, 60 fc rgb "#808080"
set object 66 rect from 0.094567, 50 to 51.981838, 60 fc rgb "#800080"
set object 67 rect from 0.095580, 50 to 52.406293, 60 fc rgb "#008080"
set object 68 rect from 0.096078, 50 to 146.271436, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
