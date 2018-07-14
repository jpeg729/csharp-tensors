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
            var t = new Tensor(new int[] { 2, 4, 5, 3 });
            t.FillWithRange_();

            t.PrintContents();
            Console.WriteLine();
            t.PrintContents(true);
            // t.Mean().PrintContents();
        }
    }
}
