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

set object 15 rect from 0.142287, 0 to 159.765248, 10 fc rgb "#FF0000"
set object 16 rect from 0.119911, 0 to 133.410975, 10 fc rgb "#00FF00"
set object 17 rect from 0.122173, 0 to 134.148089, 10 fc rgb "#0000FF"
set object 18 rect from 0.122612, 0 to 134.956395, 10 fc rgb "#FFFF00"
set object 19 rect from 0.123351, 0 to 136.154615, 10 fc rgb "#FF00FF"
set object 20 rect from 0.124440, 0 to 137.580650, 10 fc rgb "#808080"
set object 21 rect from 0.125745, 0 to 138.184142, 10 fc rgb "#800080"
set object 22 rect from 0.126535, 0 to 138.782156, 10 fc rgb "#008080"
set object 23 rect from 0.126827, 0 to 155.262613, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.139072, 10 to 156.515596, 20 fc rgb "#FF0000"
set object 25 rect from 0.089429, 10 to 100.139007, 20 fc rgb "#00FF00"
set object 26 rect from 0.091794, 10 to 100.908978, 20 fc rgb "#0000FF"
set object 27 rect from 0.092247, 10 to 101.766571, 20 fc rgb "#FFFF00"
set object 28 rect from 0.093062, 10 to 103.075414, 20 fc rgb "#FF00FF"
set object 29 rect from 0.094237, 10 to 104.518974, 20 fc rgb "#808080"
set object 30 rect from 0.095558, 10 to 105.176133, 20 fc rgb "#800080"
set object 31 rect from 0.096415, 10 to 105.820148, 20 fc rgb "#008080"
set object 32 rect from 0.096735, 10 to 151.580331, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.137241, 20 to 154.525500, 30 fc rgb "#FF0000"
set object 34 rect from 0.073018, 20 to 82.098893, 30 fc rgb "#00FF00"
set object 35 rect from 0.075315, 20 to 82.776863, 30 fc rgb "#0000FF"
set object 36 rect from 0.075690, 20 to 83.681551, 30 fc rgb "#FFFF00"
set object 37 rect from 0.076539, 20 to 85.021061, 30 fc rgb "#FF00FF"
set object 38 rect from 0.077744, 20 to 86.363856, 30 fc rgb "#808080"
set object 39 rect from 0.078982, 20 to 87.118494, 30 fc rgb "#800080"
set object 40 rect from 0.079922, 20 to 87.857798, 30 fc rgb "#008080"
set object 41 rect from 0.080351, 20 to 149.707428, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.133831, 30 to 150.946173, 40 fc rgb "#FF0000"
set object 43 rect from 0.055635, 30 to 63.281141, 40 fc rgb "#00FF00"
set object 44 rect from 0.058247, 30 to 64.077398, 40 fc rgb "#0000FF"
set object 45 rect from 0.058617, 30 to 65.070804, 40 fc rgb "#FFFF00"
set object 46 rect from 0.059547, 30 to 66.527507, 40 fc rgb "#FF00FF"
set object 47 rect from 0.060871, 30 to 67.884541, 40 fc rgb "#808080"
set object 48 rect from 0.062097, 30 to 68.555939, 40 fc rgb "#800080"
set object 49 rect from 0.062978, 30 to 69.353292, 40 fc rgb "#008080"
set object 50 rect from 0.063450, 30 to 145.127028, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.140707, 40 to 158.319499, 50 fc rgb "#FF0000"
set object 52 rect from 0.104431, 40 to 116.305122, 50 fc rgb "#00FF00"
set object 53 rect from 0.106549, 40 to 117.037856, 50 fc rgb "#0000FF"
set object 54 rect from 0.106972, 40 to 117.975403, 50 fc rgb "#FFFF00"
set object 55 rect from 0.107848, 40 to 119.363103, 50 fc rgb "#FF00FF"
set object 56 rect from 0.109109, 40 to 120.668660, 50 fc rgb "#808080"
set object 57 rect from 0.110301, 40 to 121.335677, 50 fc rgb "#800080"
set object 58 rect from 0.111155, 40 to 122.079361, 50 fc rgb "#008080"
set object 59 rect from 0.111580, 40 to 153.500331, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.135638, 50 to 156.063252, 60 fc rgb "#FF0000"
set object 61 rect from 0.032220, 50 to 38.819486, 60 fc rgb "#00FF00"
set object 62 rect from 0.035900, 50 to 40.110804, 60 fc rgb "#0000FF"
set object 63 rect from 0.036749, 50 to 42.420718, 60 fc rgb "#FFFF00"
set object 64 rect from 0.038895, 50 to 44.558676, 60 fc rgb "#FF00FF"
set object 65 rect from 0.040815, 50 to 46.468819, 60 fc rgb "#808080"
set object 66 rect from 0.042573, 50 to 47.320935, 60 fc rgb "#800080"
set object 67 rect from 0.043983, 50 to 49.021882, 60 fc rgb "#008080"
set object 68 rect from 0.044896, 50 to 147.910097, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
