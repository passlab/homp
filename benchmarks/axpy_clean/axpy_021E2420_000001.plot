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

set object 15 rect from 0.165575, 0 to 157.309773, 10 fc rgb "#FF0000"
set object 16 rect from 0.126007, 0 to 116.624355, 10 fc rgb "#00FF00"
set object 17 rect from 0.127730, 0 to 117.166260, 10 fc rgb "#0000FF"
set object 18 rect from 0.128103, 0 to 117.677051, 10 fc rgb "#FFFF00"
set object 19 rect from 0.128681, 0 to 118.586032, 10 fc rgb "#FF00FF"
set object 20 rect from 0.129684, 0 to 122.542341, 10 fc rgb "#808080"
set object 21 rect from 0.134013, 0 to 123.039410, 10 fc rgb "#800080"
set object 22 rect from 0.134768, 0 to 123.500754, 10 fc rgb "#008080"
set object 23 rect from 0.135027, 0 to 151.086038, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.159287, 10 to 153.622588, 20 fc rgb "#FF0000"
set object 25 rect from 0.047321, 10 to 45.745696, 20 fc rgb "#00FF00"
set object 26 rect from 0.050384, 10 to 46.838672, 20 fc rgb "#0000FF"
set object 27 rect from 0.051291, 10 to 48.218164, 20 fc rgb "#FFFF00"
set object 28 rect from 0.052824, 10 to 49.508871, 20 fc rgb "#FF00FF"
set object 29 rect from 0.054206, 10 to 53.737060, 20 fc rgb "#808080"
set object 30 rect from 0.058932, 10 to 54.392480, 20 fc rgb "#800080"
set object 31 rect from 0.059952, 10 to 55.080841, 20 fc rgb "#008080"
set object 32 rect from 0.060309, 10 to 144.847654, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.162722, 20 to 157.304289, 30 fc rgb "#FF0000"
set object 34 rect from 0.088564, 20 to 82.397000, 30 fc rgb "#00FF00"
set object 35 rect from 0.090336, 20 to 82.962708, 30 fc rgb "#0000FF"
set object 36 rect from 0.090738, 20 to 85.414128, 30 fc rgb "#FFFF00"
set object 37 rect from 0.093435, 20 to 86.936421, 30 fc rgb "#FF00FF"
set object 38 rect from 0.095105, 20 to 90.813098, 30 fc rgb "#808080"
set object 39 rect from 0.099340, 20 to 91.400779, 30 fc rgb "#800080"
set object 40 rect from 0.100212, 20 to 92.017747, 30 fc rgb "#008080"
set object 41 rect from 0.100642, 20 to 148.362751, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.166898, 30 to 160.715024, 40 fc rgb "#FF0000"
set object 43 rect from 0.141848, 30 to 131.066456, 40 fc rgb "#00FF00"
set object 44 rect from 0.143507, 30 to 131.612944, 40 fc rgb "#0000FF"
set object 45 rect from 0.143884, 30 to 133.884939, 40 fc rgb "#FFFF00"
set object 46 rect from 0.146408, 30 to 135.319361, 40 fc rgb "#FF00FF"
set object 47 rect from 0.147936, 30 to 139.101755, 40 fc rgb "#808080"
set object 48 rect from 0.152085, 30 to 139.650071, 40 fc rgb "#800080"
set object 49 rect from 0.152900, 30 to 140.210295, 40 fc rgb "#008080"
set object 50 rect from 0.153280, 30 to 152.365751, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.161157, 40 to 156.085890, 50 fc rgb "#FF0000"
set object 52 rect from 0.068994, 40 to 64.783976, 50 fc rgb "#00FF00"
set object 53 rect from 0.071101, 40 to 65.396374, 50 fc rgb "#0000FF"
set object 54 rect from 0.071547, 40 to 67.704994, 50 fc rgb "#FFFF00"
set object 55 rect from 0.074109, 40 to 69.565977, 50 fc rgb "#FF00FF"
set object 56 rect from 0.076131, 40 to 73.423434, 50 fc rgb "#808080"
set object 57 rect from 0.080340, 40 to 74.023011, 50 fc rgb "#800080"
set object 58 rect from 0.081254, 40 to 74.979597, 50 fc rgb "#008080"
set object 59 rect from 0.082043, 40 to 146.929257, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.164166, 50 to 158.105266, 60 fc rgb "#FF0000"
set object 61 rect from 0.108128, 50 to 100.232458, 60 fc rgb "#00FF00"
set object 62 rect from 0.109896, 50 to 100.875070, 60 fc rgb "#0000FF"
set object 63 rect from 0.110306, 50 to 103.078413, 60 fc rgb "#FFFF00"
set object 64 rect from 0.112720, 50 to 104.453322, 60 fc rgb "#FF00FF"
set object 65 rect from 0.114219, 50 to 108.245782, 60 fc rgb "#808080"
set object 66 rect from 0.118369, 50 to 108.737354, 60 fc rgb "#800080"
set object 67 rect from 0.119129, 50 to 109.271020, 60 fc rgb "#008080"
set object 68 rect from 0.119483, 50 to 149.839281, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
