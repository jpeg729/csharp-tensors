using System;
using Xunit;
using Tensors;
using System.Collections.Generic;

namespace Tensors.Test
{
    public class TensorTestsWithShape2453
    {
        Tensor t;
        public TensorTestsWithShape2453() => t = new Tensor(new int[] { 2, 4, 5, 3 });

        [Fact]
        public void TestCreateWithSizes()
        {
            Assert.True(t.size == 2 * 4 * 5 * 3);
            Assert.True(t.rank == 4);
            Assert.True(t.shapeStr == "(2,4,5,3)");
            Assert.True(t.ContentsToString(true) == "0,0:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;0,1:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;0,2:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;0,3:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;1,0:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;1,1:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;1,2:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0;1,3:;0,0,0;0,0,0;0,0,0;0,0,0;0,0,0");
        }

        [Fact]
        public void TestFillValue()
        {
            t.Fill_(5);
            Assert.True(t.ContentsToString(true) == "0,0:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;0,1:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;0,2:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;0,3:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;1,0:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;1,1:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;1,2:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5;1,3:;5,5,5;5,5,5;5,5,5;5,5,5;5,5,5");
        }

        [Fact]
        public void TestFillRange()
        {
            t.FillWithRange_();
            Assert.True(t.ContentsToString(true) == "0,0:;0,1,2;3,4,5;6,7,8;9,10,11;12,13,14;0,1:;15,16,17;18,19,20;21,22,23;24,25,26;27,28,29;0,2:;30,31,32;33,34,35;36,37,38;39,40,41;42,43,44;0,3:;45,46,47;48,49,50;51,52,53;54,55,56;57,58,59;1,0:;60,61,62;63,64,65;66,67,68;69,70,71;72,73,74;1,1:;75,76,77;78,79,80;81,82,83;84,85,86;87,88,89;1,2:;90,91,92;93,94,95;96,97,98;99,100,101;102,103,104;1,3:;105,106,107;108,109,110;111,112,113;114,115,116;117,118,119");
        }

        [Fact]
        public void TestMean()
        {
            Assert.True(t.Mean().Current == 0);
            t.Fill_(5);
            Assert.True(t.Mean().Current == 5);
            t.FillWithRange_();
            Assert.True(t.Mean().Current == 59.5);
        }

        [Fact]
        public void TestTranspose()
        {
            t.FillWithRange_();
            Assert.True(t.T().ContentsToString(true) == "0,0:;0,3,6,9,12;1,4,7,10,13;2,5,8,11,14;0,1:;15,18,21,24,27;16,19,22,25,28;17,20,23,26,29;0,2:;30,33,36,39,42;31,34,37,40,43;32,35,38,41,44;0,3:;45,48,51,54,57;46,49,52,55,58;47,50,53,56,59;1,0:;60,63,66,69,72;61,64,67,70,73;62,65,68,71,74;1,1:;75,78,81,84,87;76,79,82,85,88;77,80,83,86,89;1,2:;90,93,96,99,102;91,94,97,100,103;92,95,98,101,104;1,3:;105,108,111,114,117;106,109,112,115,118;107,110,113,116,119");
            Assert.True(t.T().T().CloseTo(t));
        }

        [Fact]
        public void TestUnPermute()
        {
            t.FillWithRange_();
            for (var n = 0; n < (t.rank ^ 2); n++) {
                var order = new int[t.rank];
                for (var i = 0; i < t.rank; i++)
                    order[i] = i;
                var rng = new Random();
                for (var i = 0; i < t.rank - 1; i++) {
                    var j = rng.Next(i, t.rank);
                    var tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
                var t2 = t.Permute(order).Permute(Tensor.UnPermuteOrder(order));
                Assert.True(t.ContentsToString(true) == t2.ContentsToString(true));
            }
        }
    }

    public class TensorTestsWithShape245
    {
        Tensor t;
        public TensorTestsWithShape245() => t = new Tensor(new int[] { 2, 4, 5 });

        [Fact]
        public void TestCreateWithSizes()
        {
            Assert.True(t.size == 2 * 4 * 5);
            Assert.True(t.rank == 3);
            Assert.True(t.shapeStr == "(2,4,5)");
            Assert.True(t.ContentsToString(true) == "0:;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;1:;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0");
        }

        [Fact]
        public void TestFillValue()
        {
            t.Fill_(5);
            Assert.True(t.ContentsToString(true) == "0:;5,5,5,5,5;5,5,5,5,5;5,5,5,5,5;5,5,5,5,5;1:;5,5,5,5,5;5,5,5,5,5;5,5,5,5,5;5,5,5,5,5");
        }

        [Fact]
        public void TestFillRange()
        {
            t.FillWithRange_();
            Assert.True(t.ContentsToString(true) == "0:;0,1,2,3,4;5,6,7,8,9;10,11,12,13,14;15,16,17,18,19;1:;20,21,22,23,24;25,26,27,28,29;30,31,32,33,34;35,36,37,38,39");
        }

        [Fact]
        public void TestMean()
        {
            Assert.True(t.Mean().Current == 0);
            t.Fill_(5);
            Assert.True(t.Mean().Current == 5);
            t.FillWithRange_();
            Assert.True(t.Mean().Current == 19.5);
        }

        [Fact]
        public void TestTranspose()
        {
            t.FillWithRange_();
            Assert.True(t.T().ContentsToString(true) == "0:;0,5,10,15;1,6,11,16;2,7,12,17;3,8,13,18;4,9,14,19;1:;20,25,30,35;21,26,31,36;22,27,32,37;23,28,33,38;24,29,34,39");
            Assert.True(t.T().T().CloseTo(t));
        }

        [Fact]
        public void TestPermute()
        {
            t.FillWithRange_();
            t = t.Permute(1, 0, 2);
            Assert.True(t.ContentsToString(true) == "0:;0,1,2,3,4;20,21,22,23,24;1:;5,6,7,8,9;25,26,27,28,29;2:;10,11,12,13,14;30,31,32,33,34;3:;15,16,17,18,19;35,36,37,38,39");
            t = t.Permute(Tensor.UnPermuteOrder(1, 0, 2));
            Assert.True(t.ContentsToString(true) == "0:;0,1,2,3,4;5,6,7,8,9;10,11,12,13,14;15,16,17,18,19;1:;20,21,22,23,24;25,26,27,28,29;30,31,32,33,34;35,36,37,38,39");
            t = t.Permute(1, 2, 0);
            Assert.True(t.ContentsToString(true) == "0:;0,20;1,21;2,22;3,23;4,24;1:;5,25;6,26;7,27;8,28;9,29;2:;10,30;11,31;12,32;13,33;14,34;3:;15,35;16,36;17,37;18,38;19,39");
        }

        [Fact]
        public void TestUnPermute()
        {
            t.FillWithRange_();
            for (var n = 0; n < (t.rank ^ 2); n++) {
                var order = new int[t.rank];
                for (var i = 0; i < t.rank; i++)
                    order[i] = i;
                var rng = new Random();
                for (var i = 0; i < t.rank - 1; i++) {
                    var j = rng.Next(i, t.rank);
                    var tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
                var t2 = t.Permute(order).Permute(Tensor.UnPermuteOrder(order));
                Assert.True(t.ContentsToString(true) == t2.ContentsToString(true));
            }
        }
    }

    public class TensorFromDataTests
    {
        [Fact]
        public void TestCreate()
        {
            var d = new double[] { 1, 2, 5, 5 };
            var t = new Tensor(d);
            Assert.True(t.ContentsToString(true) == "1,2,5,5");
        }
    }
}
