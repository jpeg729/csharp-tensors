using System;
using System.Text;
using System.Linq;
using System.Globalization;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Tensors
{
    public static class Initialisers
    {
        public static void FillWithRange_(this Tensor t, double start = 0, double step = 1)
        {
            t.WarnAboutInplaceModification();
            t.Reset();
            while (t.MoveNext()) {
                t.SetCurrent(start); start += step;
            }
        }

        public static void FillEye_(this Tensor t, int dim1, int dim2)
        {
            // TODO
        }

        public static void FillUniform_(this Tensor t, double minval = 0, double maxval = 1)
        {
            t.WarnAboutInplaceModification();
            t.Reset();
            while (t.MoveNext()) {
                t.SetCurrent(minval + Tensor._rng.NextDouble() * (maxval - minval));
            }
        }

        public static void FillNormal_(this Tensor t, double mean = 0, double std = 1)
        {
            t.WarnAboutInplaceModification();

            // increase std to compensate for truncation at 2*std
            // .8796 = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            std /= .8796;

            while (t.MoveNext()) {
                var distance = Math.Sqrt(-2.0 * Math.Log(Tensor._rng.NextDouble()));
                var angle = 2.0 * Math.PI * Tensor._rng.NextDouble();

                // Discard values that are more than 2*std away from the mean.
                var randomBit = distance * Math.Sin(angle);
                if (Math.Abs(randomBit) <= 2) {
                    t.SetCurrent(mean + std * randomBit);
                    if (!t.MoveNext())
                        break;
                }
                // The Box-Muller transform gives two values each time
                randomBit = distance * Math.Cos(angle);
                if (Math.Abs(randomBit) <= 2)
                    t.SetCurrent(mean + std * randomBit);
            }
        }

        public static void InitialiseWeights_(this Tensor t, Distribution dist, Activation activation, int fan, double activationParam = 0)
        {
            // Choosing fan_in preserves the magnitude of the variance of the weights in the forward pass. 
            // Choosing fan_out preserves the magnitudes in the backwards pass.
            double gain = 1;
            if (activation == Activation.ReLU)
                gain = Math.Sqrt(2 / (1 + activationParam));
            else if (activation == Activation.Tanh)
                gain = 1.1;

            if (dist == Distribution.Uniform) {
                var limit = Math.Sqrt(3.0 / fan);
                Console.WriteLine($"InitialiseWeights using limit {limit}");
                t.FillUniform_(-limit, limit);
            } else if (dist == Distribution.Normal) {
                var std = Math.Sqrt(1.0 / fan);
                Console.WriteLine($"InitialiseWeights using std {std}");
                t.FillNormal_(0, std);
            }
        }
    }
}