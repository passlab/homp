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

set object 15 rect from 0.279473, 0 to 156.117061, 10 fc rgb "#FF0000"
set object 16 rect from 0.088680, 0 to 47.373665, 10 fc rgb "#00FF00"
set object 17 rect from 0.090935, 0 to 47.727191, 10 fc rgb "#0000FF"
set object 18 rect from 0.091372, 0 to 48.153930, 10 fc rgb "#FFFF00"
set object 19 rect from 0.092223, 0 to 48.771545, 10 fc rgb "#FF00FF"
set object 20 rect from 0.093435, 0 to 57.439664, 10 fc rgb "#808080"
set object 21 rect from 0.109956, 0 to 57.740363, 10 fc rgb "#800080"
set object 22 rect from 0.110835, 0 to 58.087608, 10 fc rgb "#008080"
set object 23 rect from 0.111202, 0 to 145.796379, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.284963, 10 to 158.702067, 20 fc rgb "#FF0000"
set object 25 rect from 0.207497, 10 to 109.263220, 20 fc rgb "#00FF00"
set object 26 rect from 0.209261, 10 to 109.578039, 20 fc rgb "#0000FF"
set object 27 rect from 0.209642, 10 to 109.909076, 20 fc rgb "#FFFF00"
set object 28 rect from 0.210290, 10 to 110.429944, 20 fc rgb "#FF00FF"
set object 29 rect from 0.211281, 10 to 119.023797, 20 fc rgb "#808080"
set object 30 rect from 0.227730, 10 to 119.310384, 20 fc rgb "#800080"
set object 31 rect from 0.228538, 10 to 119.593308, 20 fc rgb "#008080"
set object 32 rect from 0.228809, 10 to 148.694116, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.281392, 20 to 166.615527, 30 fc rgb "#FF0000"
set object 34 rect from 0.118736, 20 to 62.940190, 30 fc rgb "#00FF00"
set object 35 rect from 0.120703, 20 to 63.342343, 30 fc rgb "#0000FF"
set object 36 rect from 0.121233, 20 to 64.924312, 30 fc rgb "#FFFF00"
set object 37 rect from 0.124277, 20 to 73.685507, 30 fc rgb "#FF00FF"
set object 38 rect from 0.141049, 20 to 82.268381, 30 fc rgb "#808080"
set object 39 rect from 0.157468, 20 to 82.811217, 30 fc rgb "#800080"
set object 40 rect from 0.158723, 20 to 83.157940, 30 fc rgb "#008080"
set object 41 rect from 0.159137, 20 to 146.830797, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.283311, 30 to 164.317645, 40 fc rgb "#FF0000"
set object 43 rect from 0.167538, 30 to 88.327430, 40 fc rgb "#00FF00"
set object 44 rect from 0.169229, 30 to 88.653759, 40 fc rgb "#0000FF"
set object 45 rect from 0.169628, 30 to 90.157800, 40 fc rgb "#FFFF00"
set object 46 rect from 0.172513, 30 to 95.742005, 40 fc rgb "#FF00FF"
set object 47 rect from 0.183188, 30 to 104.326967, 40 fc rgb "#808080"
set object 48 rect from 0.199616, 30 to 104.839474, 40 fc rgb "#800080"
set object 49 rect from 0.200828, 30 to 105.149586, 40 fc rgb "#008080"
set object 50 rect from 0.201176, 30 to 147.798276, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.286305, 40 to 165.908482, 50 fc rgb "#FF0000"
set object 52 rect from 0.236846, 40 to 124.502363, 50 fc rgb "#00FF00"
set object 53 rect from 0.238400, 40 to 124.772733, 50 fc rgb "#0000FF"
set object 54 rect from 0.238697, 40 to 126.237553, 50 fc rgb "#FFFF00"
set object 55 rect from 0.241517, 40 to 131.921116, 50 fc rgb "#FF00FF"
set object 56 rect from 0.252367, 40 to 140.541122, 50 fc rgb "#808080"
set object 57 rect from 0.268850, 40 to 141.038978, 50 fc rgb "#800080"
set object 58 rect from 0.270047, 40 to 141.361645, 50 fc rgb "#008080"
set object 59 rect from 0.270420, 40 to 149.474374, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.277595, 50 to 164.074994, 60 fc rgb "#FF0000"
set object 61 rect from 0.037849, 50 to 21.303429, 60 fc rgb "#00FF00"
set object 62 rect from 0.041170, 50 to 21.967590, 60 fc rgb "#0000FF"
set object 63 rect from 0.042125, 50 to 24.269158, 60 fc rgb "#FFFF00"
set object 64 rect from 0.046562, 50 to 31.362625, 60 fc rgb "#FF00FF"
set object 65 rect from 0.060087, 50 to 40.077804, 60 fc rgb "#808080"
set object 66 rect from 0.076764, 50 to 40.646270, 60 fc rgb "#800080"
set object 67 rect from 0.078267, 50 to 41.259701, 60 fc rgb "#008080"
set object 68 rect from 0.079033, 50 to 144.599310, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
