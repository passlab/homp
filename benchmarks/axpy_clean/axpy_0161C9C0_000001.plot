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

set object 15 rect from 0.187191, 0 to 150.942606, 10 fc rgb "#FF0000"
set object 16 rect from 0.105862, 0 to 85.137479, 10 fc rgb "#00FF00"
set object 17 rect from 0.108461, 0 to 86.034442, 10 fc rgb "#0000FF"
set object 18 rect from 0.109368, 0 to 86.721931, 10 fc rgb "#FFFF00"
set object 19 rect from 0.110272, 0 to 87.728356, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111549, 0 to 88.832432, 10 fc rgb "#808080"
set object 21 rect from 0.112958, 0 to 89.304145, 10 fc rgb "#800080"
set object 22 rect from 0.113799, 0 to 89.804208, 10 fc rgb "#008080"
set object 23 rect from 0.114195, 0 to 146.771215, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.192106, 10 to 154.220971, 20 fc rgb "#FF0000"
set object 25 rect from 0.155047, 10 to 123.455684, 20 fc rgb "#00FF00"
set object 26 rect from 0.157115, 10 to 124.025047, 20 fc rgb "#0000FF"
set object 27 rect from 0.157604, 10 to 124.647960, 20 fc rgb "#FFFF00"
set object 28 rect from 0.158427, 10 to 125.515786, 20 fc rgb "#FF00FF"
set object 29 rect from 0.159522, 10 to 126.521424, 20 fc rgb "#808080"
set object 30 rect from 0.160792, 10 to 126.985262, 20 fc rgb "#800080"
set object 31 rect from 0.161633, 10 to 127.441225, 20 fc rgb "#008080"
set object 32 rect from 0.161950, 10 to 150.837081, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.190473, 20 to 153.139732, 30 fc rgb "#FF0000"
set object 34 rect from 0.140445, 20 to 111.942425, 30 fc rgb "#00FF00"
set object 35 rect from 0.142519, 20 to 112.506275, 30 fc rgb "#0000FF"
set object 36 rect from 0.142981, 20 to 113.279601, 30 fc rgb "#FFFF00"
set object 37 rect from 0.143968, 20 to 114.323040, 30 fc rgb "#FF00FF"
set object 38 rect from 0.145286, 20 to 115.228665, 30 fc rgb "#808080"
set object 39 rect from 0.146437, 20 to 115.654703, 30 fc rgb "#800080"
set object 40 rect from 0.147227, 20 to 116.120116, 30 fc rgb "#008080"
set object 41 rect from 0.147573, 20 to 149.530617, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.185025, 30 to 151.258395, 40 fc rgb "#FF0000"
set object 43 rect from 0.069383, 30 to 57.022132, 40 fc rgb "#00FF00"
set object 44 rect from 0.072905, 30 to 57.978945, 40 fc rgb "#0000FF"
set object 45 rect from 0.073751, 30 to 59.842960, 40 fc rgb "#FFFF00"
set object 46 rect from 0.076197, 30 to 61.491986, 40 fc rgb "#FF00FF"
set object 47 rect from 0.078217, 30 to 62.731512, 40 fc rgb "#808080"
set object 48 rect from 0.079789, 30 to 63.331587, 40 fc rgb "#800080"
set object 49 rect from 0.081022, 30 to 64.271863, 40 fc rgb "#008080"
set object 50 rect from 0.081767, 30 to 144.899326, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.193695, 40 to 155.610121, 50 fc rgb "#FF0000"
set object 52 rect from 0.170720, 40 to 135.797394, 50 fc rgb "#00FF00"
set object 53 rect from 0.172801, 40 to 136.404557, 50 fc rgb "#0000FF"
set object 54 rect from 0.173340, 40 to 137.107007, 50 fc rgb "#FFFF00"
set object 55 rect from 0.174217, 40 to 138.007908, 50 fc rgb "#FF00FF"
set object 56 rect from 0.175376, 40 to 138.937159, 50 fc rgb "#808080"
set object 57 rect from 0.176541, 40 to 139.446672, 50 fc rgb "#800080"
set object 58 rect from 0.177448, 40 to 139.918385, 50 fc rgb "#008080"
set object 59 rect from 0.177794, 40 to 152.114407, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.188862, 50 to 152.102594, 60 fc rgb "#FF0000"
set object 61 rect from 0.123699, 50 to 98.938428, 60 fc rgb "#00FF00"
set object 62 rect from 0.125990, 50 to 99.644028, 60 fc rgb "#0000FF"
set object 63 rect from 0.126646, 50 to 100.408691, 60 fc rgb "#FFFF00"
set object 64 rect from 0.127647, 50 to 101.426142, 60 fc rgb "#FF00FF"
set object 65 rect from 0.128919, 50 to 102.405793, 60 fc rgb "#808080"
set object 66 rect from 0.130193, 50 to 102.925543, 60 fc rgb "#800080"
set object 67 rect from 0.131072, 50 to 103.451594, 60 fc rgb "#008080"
set object 68 rect from 0.131488, 50 to 148.225729, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
