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

set object 15 rect from 0.180928, 0 to 156.759536, 10 fc rgb "#FF0000"
set object 16 rect from 0.157638, 0 to 135.621719, 10 fc rgb "#00FF00"
set object 17 rect from 0.159882, 0 to 136.145361, 10 fc rgb "#0000FF"
set object 18 rect from 0.160283, 0 to 136.798212, 10 fc rgb "#FFFF00"
set object 19 rect from 0.161039, 0 to 137.670384, 10 fc rgb "#FF00FF"
set object 20 rect from 0.162067, 0 to 138.743173, 10 fc rgb "#808080"
set object 21 rect from 0.163343, 0 to 139.182658, 10 fc rgb "#800080"
set object 22 rect from 0.164100, 0 to 139.611943, 10 fc rgb "#008080"
set object 23 rect from 0.164354, 0 to 153.311656, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.179303, 10 to 155.784505, 20 fc rgb "#FF0000"
set object 25 rect from 0.140220, 10 to 120.998830, 20 fc rgb "#00FF00"
set object 26 rect from 0.142704, 10 to 121.712889, 20 fc rgb "#0000FF"
set object 27 rect from 0.143293, 10 to 122.361491, 20 fc rgb "#FFFF00"
set object 28 rect from 0.144097, 10 to 123.394324, 20 fc rgb "#FF00FF"
set object 29 rect from 0.145302, 10 to 124.524067, 20 fc rgb "#808080"
set object 30 rect from 0.146613, 10 to 125.033259, 20 fc rgb "#800080"
set object 31 rect from 0.147505, 10 to 125.579853, 20 fc rgb "#008080"
set object 32 rect from 0.147849, 10 to 151.855488, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.177523, 20 to 154.761023, 30 fc rgb "#FF0000"
set object 34 rect from 0.077014, 20 to 103.427902, 30 fc rgb "#00FF00"
set object 35 rect from 0.122148, 20 to 104.232920, 30 fc rgb "#0000FF"
set object 36 rect from 0.122729, 20 to 105.139941, 30 fc rgb "#FFFF00"
set object 37 rect from 0.123800, 20 to 106.432895, 30 fc rgb "#FF00FF"
set object 38 rect from 0.125320, 20 to 107.459782, 30 fc rgb "#808080"
set object 39 rect from 0.126532, 20 to 107.961320, 30 fc rgb "#800080"
set object 40 rect from 0.127380, 20 to 108.609925, 30 fc rgb "#008080"
set object 41 rect from 0.127881, 20 to 150.487726, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.175732, 30 to 152.549143, 40 fc rgb "#FF0000"
set object 43 rect from 0.060783, 30 to 53.130168, 40 fc rgb "#00FF00"
set object 44 rect from 0.062937, 30 to 53.774520, 40 fc rgb "#0000FF"
set object 45 rect from 0.063373, 30 to 54.502177, 40 fc rgb "#FFFF00"
set object 46 rect from 0.064229, 30 to 55.461057, 40 fc rgb "#FF00FF"
set object 47 rect from 0.065361, 30 to 56.439486, 40 fc rgb "#808080"
set object 48 rect from 0.066517, 30 to 56.965677, 40 fc rgb "#800080"
set object 49 rect from 0.067399, 30 to 57.492720, 40 fc rgb "#008080"
set object 50 rect from 0.067753, 30 to 148.734883, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.173798, 40 to 151.130377, 50 fc rgb "#FF0000"
set object 52 rect from 0.044612, 40 to 39.611520, 50 fc rgb "#00FF00"
set object 53 rect from 0.046963, 40 to 40.206566, 50 fc rgb "#0000FF"
set object 54 rect from 0.047412, 40 to 41.008182, 50 fc rgb "#FFFF00"
set object 55 rect from 0.048354, 40 to 42.061419, 50 fc rgb "#FF00FF"
set object 56 rect from 0.049603, 40 to 43.080651, 50 fc rgb "#808080"
set object 57 rect from 0.050803, 40 to 43.621295, 50 fc rgb "#800080"
set object 58 rect from 0.051699, 40 to 44.186590, 50 fc rgb "#008080"
set object 59 rect from 0.052116, 40 to 147.077249, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.171486, 50 to 151.118480, 60 fc rgb "#FF0000"
set object 61 rect from 0.024119, 50 to 22.827755, 60 fc rgb "#00FF00"
set object 62 rect from 0.027306, 50 to 23.722028, 60 fc rgb "#0000FF"
set object 63 rect from 0.028049, 50 to 25.568378, 60 fc rgb "#FFFF00"
set object 64 rect from 0.030234, 50 to 27.055147, 60 fc rgb "#FF00FF"
set object 65 rect from 0.031951, 50 to 28.259698, 60 fc rgb "#808080"
set object 66 rect from 0.033381, 50 to 28.917651, 60 fc rgb "#800080"
set object 67 rect from 0.034624, 50 to 29.897781, 60 fc rgb "#008080"
set object 68 rect from 0.035312, 50 to 144.991178, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
