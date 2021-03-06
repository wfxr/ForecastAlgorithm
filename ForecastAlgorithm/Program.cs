﻿using System;
using System.Linq;
using Wfxr.DataAnalysis.Clustering;
using Wfxr.Utility.ContainerExtension;

namespace ForecastAlgorithm {
    internal class Program {
        //private static void Main() {
        //    var data = new double[,] {
        //        #region 1
        //        {20027},
        //        {26994},
        //        {22766},
        //        {16287},
        //        {22854},
        //        {17559},
        //        {15029},
        //        {20287},
        //        {15933},
        //        {22003},
        //        {19271},
        //        {20619},
        //        {28489},
        //        {22026},
        //        {13946},
        //        {21626},
        //        {15905},
        //        {18525},
        //        {16023},
        //        {15106},
        //        {14082},
        //        {24273},
        //        {15566},
        //        {16031},
        //        {16307},
        //        {18313},
        //        {17281},
        //        {20802},
        //        {18310},
        //        {17624},
        //        {13872},
        //        {15265},
        //        {19849},
        //        {16000},
        //        {21441},
        //        {17046},
        //        {20483},
        //        {21946},
        //        {21323},
        //        {14246},
        //        {22336},
        //        {13559},
        //        {17994},
        //        {17838},
        //        {16351},
        //        {24887},
        //        {22318},
        //        {21395},
        //        {21194},
        //        {15981},
        //        {21142},
        //        {19759},
        //        {21192},
        //        {13294},
        //        {14797},
        //        {18825},
        //        {17314},
        //        {16402},
        //        {13043},
        //        {20700},
        //        #endregion
        //        #region data2
        //        //{4.0},
        //        //{9.1},
        //        //{9.05},
        //        //{8.8},
        //        //{0},
        //        //{4.1},
        //        //{8.9},
        //        //{9.2},
        //        //{9.0},
        //        //{3.8},
        //        //{0.1},
        //        //{4.4},
        //        //{9.32},
        //        //{9.3}
        //        #endregion
        //    };
        //    //var data1 = new double[,] {

        //    #region 2

        //    //    {583, 495, 487, 650, 730, 1960, 5620, 2800, 3370, 2640, 1090, 680},
        //    //    {528, 491, 474, 558, 642, 1880, 5750, 7070, 4580, 3530, 1620, 890},
        //    //    {609, 525, 529, 498, 821, 2430, 5650, 5360, 2990, 2430, 1330, 728},
        //    //    {564, 479, 482, 508, 967, 1890, 2810, 2380, 3450, 1940, 1170, 690},
        //    //    {561, 467, 461, 858, 895, 2220, 4870, 4530, 3270, 3550, 1390, 810},
        //    //    {621, 530, 505, 433, 858, 1310, 2480, 4300, 3760, 2070, 1130, 713},
        //    //    {529, 466, 465, 555, 670, 2460, 2290, 3250, 1940, 1710, 1040, 649},
        //    //    {460, 425, 409, 515, 544, 2820, 3740, 3850, 4370, 2160, 1170, 709},
        //    //    {530, 468, 463, 566, 999, 1770, 3860, 2870, 2190, 1740, 909, 566},
        //    //    {428, 400, 383, 473, 875, 2490, 3890, 6190, 4070, 1900, 1050, 682},
        //    //    {481, 426, 437, 440, 566, 1860, 3950, 4050, 3390, 2600, 1250, 728},
        //    //    {524, 461, 456, 493, 826, 2150, 2900, 3890, 4840, 3080, 1220, 764},
        //    //    {552, 482, 440, 527, 768, 3630, 6110, 5900, 5710, 2940, 1520, 944},
        //    //    {667, 558, 534, 675, 711, 1870, 3640, 4560, 5260, 2690, 1270, 816},
        //    //    {591, 503, 484, 480, 762, 1540, 2550, 2300, 2250, 2010, 968, 602},
        //    //    {462, 410, 427, 566, 1090, 2880, 4010, 3890, 4470, 2490, 1090, 713},
        //    //    {528, 467, 450, 711, 455, 989, 3180, 2750, 4030, 1780, 941, 619},
        //    //    {463, 411, 404, 545, 890, 1560, 4420, 3660, 2610, 2500, 1200, 736},
        //    //    {531, 462, 457, 527, 958, 2210, 2880, 2660, 2310, 2250, 1130, 641},
        //    //    {499, 439, 442, 476, 861, 1990, 3130, 2680, 2330, 1710, 903, 584},
        //    //    {462, 447, 463, 482, 732, 1780, 1970, 2460, 2910, 1710, 976, 599},
        //    //    {434, 392, 395, 615, 956, 3000, 5050, 4490, 5190, 2490, 1290, 797},
        //    //    {568, 498, 469, 627, 720, 1750, 3340, 2710, 2480, 1840, 1030, 600},
        //    //    {446, 398, 398, 529, 889, 1940, 3480, 2250, 2880, 1990, 1020, 655},
        //    //    {475, 418, 429, 465, 956, 2070, 2560, 3380, 2520, 2200, 1060, 667},
        //    //    {491, 423, 397, 564, 1160, 2670, 3310, 3080, 3180, 2140, 1120, 692},
        //    //    {501, 430, 400, 500, 773, 1200, 2400, 3260, 3820, 2920, 1230, 778},
        //    //    {530, 457, 448, 533, 800, 1550, 2540, 4670, 4250, 3610, 1540, 861},
        //    //    {596, 521, 488, 522, 927, 2720, 4270, 2680, 3490, 1670, 925, 618},
        //    //    {482, 416, 397, 443, 640, 1710, 5010, 2450, 3330, 1980, 1010, 654},
        //    //    {476, 411, 423, 471, 1040, 1260, 2320, 2620, 2640, 1640, 898, 560},
        //    //    {427, 386, 378, 607, 682, 1560, 4130, 2640, 2460, 1350, 874, 584},
        //    //    {425, 369, 391, 571, 807, 2420, 4500, 3110, 3750, 2460, 1130, 710},
        //    //    {528, 451, 431, 490, 880, 932, 2040, 2260, 4270, 2830, 1160, 707},
        //    //    {513, 453, 420, 481, 557, 973, 4470, 4240, 5220, 2940, 1330, 810},
        //    //    {591, 498, 469, 547, 701, 1390, 2360, 2570, 4360, 2610, 1270, 769},
        //    //    {531, 469, 445, 500, 862, 1770, 3840, 2950, 4000, 3730, 1520, 866},
        //    //    {599, 524, 487, 533, 1180, 3040, 3330, 3410, 4340, 3420, 1370, 836},
        //    //    {582, 507, 485, 572, 939, 1680, 3760, 5130, 4270, 2500, 1210, 777},
        //    //    {562, 486, 486, 458, 932, 1380, 3480, 2140, 2200, 1610, 969, 591},
        //    //    {462, 418, 405, 511, 784, 1280, 3370, 6370, 4530, 2870, 1370, 846},
        //    //    {613, 516, 496, 568, 866, 2470, 2130, 1790, 2320, 1530, 831, 558},
        //    //    {423, 373, 384, 683, 1060, 1930, 4320, 2790, 3140, 1950, 1060, 677},
        //    //    {491, 451, 453, 486, 812, 1490, 4720, 3380, 2790, 2010, 1040, 657},
        //    //    {477, 431, 437, 550, 957, 2170, 3950, 2350, 2620, 1800, 922, 595},
        //    //    {456, 404, 404, 550, 771, 1710, 5880, 6920, 4410, 2060, 1320, 862},
        //    //    {619, 515, 518, 483, 956, 3595, 5450, 3523, 3744, 1942, 1380, 727},
        //    //    {533, 466, 433, 612, 1275, 3009, 3926, 3904, 4574, 1870, 1093, 699},
        //    //    {520, 484, 435, 559, 867, 3455, 2485, 3679, 5266, 2522, 1171, 755},
        //    //    {532, 477, 442, 521, 716, 2058, 3398, 3755, 1788, 1798, 907, 598},
        //    //    {464, 403, 365, 431, 548, 2702, 4066, 3495, 5565, 2111, 1127, 731.78},
        //    //    {528, 445, 473, 578, 895, 2379, 3657, 3163, 4115, 2570.7, 1185, 743},
        //    //    {552, 475, 502, 565, 1044, 2157, 3584, 5574, 3687, 2183, 1166, 730},
        //    //    {553, 484, 455, 498, 782, 2646, 2562, 1378, 1834, 1749, 835, 555},
        //    //    {439, 371, 375, 435, 897, 1640, 2818, 2293, 3219, 1590, 935, 595},
        //    //{470, 391, 389, 505, 1108, 2030, 3089, 4108, 3995, 1840, 1088, 673},
        //    //{498, 448, 412, 501, 567, 1387, 3915, 4924, 2370, 1663, 943, 632},

        //    #endregion

        //    //};

        //    var fcmmc = new FcmMc(data);
        //    fcmmc.Train(3, 2);

        //    Console.WriteLine("聚类中心值：");
        //    var center = fcmmc.Center;
        //    for (var i = 0; i < center.GetLength(0); ++i) {
        //        for (var j = 0; j < center.GetLength(1); ++j)
        //            Console.Write($"{center[i, j]:F0}  ");
        //        Console.WriteLine();
        //    }
        //    Console.WriteLine();

        //    var result = fcmmc.Forecast();

        //    Console.WriteLine("预测值：");
        //    result.ForEach(i => Console.Write($"{i:F0}  "));
        //    Console.WriteLine();

        //    Console.WriteLine();
        //}
        private static double[,] _actual;
        private static double[,] _forecast;
        private static double[,] _data;

        private static void Main() {
            #region data

            _data = new double[,] {
{583,495,487,650,730,1960,5620,2800,3370,2640,1090,680},
{528,491,474,558,642,1880,5750,7070,4580,3530,1620,890},
{609,525,529,498,821,2430,5650,5360,2990,2430,1330,728},
{564,479,482,508,967,1890,2810,2380,3450,1940,1170,690},
{561,467,461,858,895,2220,4870,4530,3270,3550,1390,810},
{621,530,505,433,858,1310,2480,4300,3760,2070,1130,713},
{529,466,465,555,670,2460,2290,3250,1940,1710,1040,649},
{460,425,409,515,544,2820,3740,3850,4370,2160,1170,709},
{530,468,463,566,999,1770,3860,2870,2190,1740,909,566},
{428,400,383,473,875,2490,3890,6190,4070,1900,1050,682},
{481,426,437,440,566,1860,3950,4050,3390,2600,1250,728},
{524,461,456,493,826,2150,2900,3890,4840,3080,1220,764},
{552,482,440,527,768,3630,6110,5900,5710,2940,1520,944},
{667,558,534,675,711,1870,3640,4560,5260,2690,1270,816},
{591,503,484,480,762,1540,2550,2300,2250,2010,968,602},
{462,410,427,566,1090,2880,4010,3890,4470,2490,1090,713},
{528,467,450,711,455,989,3180,2750,4030,1780,941,619},
{463,411,404,545,890,1560,4420,3660,2610,2500,1200,736},
{531,462,457,527,958,2210,2880,2660,2310,2250,1130,641},
{499,439,442,476,861,1990,3130,2680,2330,1710,903,584},
{462,447,463,482,732,1780,1970,2460,2910,1710,976,599},
{434,392,395,615,956,3000,5050,4490,5190,2490,1290,797},
{568,498,469,627,720,1750,3340,2710,2480,1840,1030,600},
{446,398,398,529,889,1940,3480,2250,2880,1990,1020,655},
{475,418,429,465,956,2070,2560,3380,2520,2200,1060,667},
{491,423,397,564,1160,2670,3310,3080,3180,2140,1120,692},
{501,430,400,500,773,1200,2400,3260,3820,2920,1230,778},
{530,457,448,533,800,1550,2540,4670,4250,3610,1540,861},
{596,521,488,522,927,2720,4270,2680,3490,1670,925,618},
{482,416,397,443,640,1710,5010,2450,3330,1980,1010,654},
{476,411,423,471,1040,1260,2320,2620,2640,1640,898,560},
{427,386,378,607,682,1560,4130,2640,2460,1350,874,584},
{425,369,391,571,807,2420,4500,3110,3750,2460,1130,710},
{528,451,431,490,880,932,2040,2260,4270,2830,1160,707},
{513,453,420,481,557,973,4470,4240,5220,2940,1330,810},
{591,498,469,547,701,1390,2360,2570,4360,2610,1270,769},
{531,469,445,500,862,1770,3840,2950,4000,3730,1520,866},
{599,524,487,533,1180,3040,3330,3410,4340,3420,1370,836},
{582,507,485,572,939,1680,3760,5130,4270,2500,1210,777},
{562,486,486,458,932,1380,3480,2140,2200,1610,969,591},
{462,418,405,511,784,1280,3370,6370,4530,2870,1370,846},
{613,516,496,568,866,2470,2130,1790,2320,1530,831,558},
{423,373,384,683,1060,1930,4320,2790,3140,1950,1060,677},
{491,451,453,486,812,1490,4720,3380,2790,2010,1040,657},
{477,431,437,550,957,2170,3950,2350,2620,1800,922,595},
{456,404,404,550,771,1710,5880,6920,4410,2060,1320,862},
{619,515,518,483,956,3595,5450,3523,3744,1942,1380,727},
{533,466,433,612,1275,3009,3926,3904,4574,1870,1093,699},
{520,484,435,559,867,3455,2485,3679,5266,2522,1171,755},
{532,477,442,521,716,2058,3398,3755,1788,1798,907,598},
{464,403,365,431,548,2702,4066,3495,5565,2111,1127,731.78},
{528,445,473,578,895,2379,3657,3163,4115,2570.7,1185,743},
{552,475,502,565,1044,2157,3584,5574,3687,2183,1166,730},
{553,484,455,498,782,2646,2562,1378,1834,1749,835,555},
{439,371,375,435,897,1640,2818,2293,3219,1590,935,595},
{470,391,389,505,1108,2030,3089,4108,3995,1840,1088,673},
{498,448,412,501,567,1387,3915,4924,2370,1663,943,632},
{471,409,411,556,755,1806,3316,2628,3331,1861,1078,659},
{478,426,375,552,831,1756,3040,2140,1604,1458,759,527},
{389,344,346,520,824,2243,5821,3003,3598,2458,1175,712},

            };

            #endregion

            SimForecast(15);

            Console.WriteLine("预测值：");
            PrintArray(_forecast);
            Console.WriteLine();

            Console.WriteLine("实际值：");
            PrintArray(_actual);
            Console.WriteLine();
        }

        private static void PrintArray(double[,] arr) {
            for (var i = 0; i < arr.GetLength(0); ++i) {
                for (var j = 0; j < arr.GetLength(1); ++j)
                    Console.Write("{0:F0}  ", arr[i, j]);
                Console.WriteLine();
            }
        }

        private static void SimForecast(int simForecastCount) {
            var dataCount = _data.GetLength(0);
            var dim = _data.GetLength(1);

            // 实际值
            _actual = new double[simForecastCount, dim];
            _forecast = new double[simForecastCount, dim];

            for (var iter = 0; iter < simForecastCount; ++ iter) {
                var obsCount = dataCount - simForecastCount + iter;
                var obs = new double[obsCount, dim];
                // 模拟观测值序列
                for (var i = 0; i < obsCount; ++i)
                    for (var j = 0; j < dim; ++j)
                        obs[i, j] = _data[i, j];

                var fcmmc = new FcmMc(obs);
                fcmmc.Train(3, 1);
                var result = fcmmc.WeightedForecast();
                // 实际值
                for (var j = 0; j < dim; ++j) {
                    _forecast[iter, j] = result[j];
                    _actual[iter, j] = _data[obsCount, j];
                }
            }
        }
    }
}