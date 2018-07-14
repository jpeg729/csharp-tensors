using System;
using Tensors;

namespace MnistExample
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            Tensor.Seed(0);
            var t = new Tensor(new int[] {200,400,5,3});
            Initialisers.FillWithRange_(t);
            // Utils.PrintContents(t);
            //t = t.Pad(0, 2, 0, Padding.Const);
            //Console.WriteLine("Padding done!");
            Utils.PrintContents(t);
        }
    }
}
