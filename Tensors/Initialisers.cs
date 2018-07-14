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
		public static void FillWithRange_(Tensor t, double start = 0, double step = 1)
		{
			t.WarnAboutInplaceModification();
			t.ResetOffset();
			do { t.SetItem(start); start += step; }
			while (t.AdvanceOffset());
		}

		public static void FillEye_(Tensor t, int dim1, int dim2)
		{
			// TODO
		}

		public static void FillUniform_(Tensor t, double minval = 0, double maxval = 1)
		{
			t.WarnAboutInplaceModification();
			t.ResetOffset();
			do { t.SetItem(minval + Tensor._rng.NextDouble() * (maxval - minval)); }
			while (t.AdvanceOffset());
		}

		public static void FillNormal_(Tensor t, double mean = 0, double std = 1)
		{
			t.WarnAboutInplaceModification();

			// increase std to compensate for truncation at 2*std
			// .8796 = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
			std /= .8796;

			do // The Box-Muller transform gives two values each time
			{
				var distance = Math.Sqrt(-2.0 * Math.Log(Tensor._rng.NextDouble()));
				var angle = 2.0 * Math.PI * Tensor._rng.NextDouble();

				// Discard values that are more than 2*std away from the mean.
				var randomBit = distance * Math.Sin(angle);
				if (Math.Abs(randomBit) <= 2)
				{
					t.SetItem(mean + std * randomBit);
					if (!t.AdvanceOffset())
						break;
				}
				randomBit = distance * Math.Cos(angle);
				if (Math.Abs(randomBit) <= 2)
					t.SetItem(mean + std * randomBit);
			} while (t.AdvanceOffset());
		}

		public static void InitialiseWeights_(Tensor t, Distribution dist, Activation activation, int fan, double activationParam = 0)
		{
			// Choosing fan_in preserves the magnitude of the variance of the weights in the forward pass. 
			// Choosing fan_out preserves the magnitudes in the backwards pass.
			double gain = 1;
			if (activation == Activation.ReLU)
				gain = Math.Sqrt(2 / (1 + activationParam));
			else if (activation == Activation.Tanh)
				gain = 1.1;

			if (dist == Distribution.Uniform)
			{
				var limit = Math.Sqrt(3.0 / fan);
				Console.WriteLine($"InitialiseWeights using limit {limit}");
				FillUniform_(t, -limit, limit);
			}
			else if (dist == Distribution.Normal)
			{
				var std = Math.Sqrt(1.0 / fan);
				Console.WriteLine($"InitialiseWeights using std {std}");
				FillNormal_(t, 0, std);
			}
		}
    }
}