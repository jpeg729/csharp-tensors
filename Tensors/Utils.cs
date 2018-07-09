using System;
using System.Text;
using System.Linq;
using System.Globalization;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Tensors
{
    
    class Utils
    {
        public void PrintContents(Tensor t)
        {
            Console.Error.WriteLine(t);
            Console.Error.WriteLine(ContentsToString(t));
        }

        public string ContentsToString(Tensor t)
        {
            t.ResetOffset();
            var output = new StringBuilder(t.size * 10);
            var indices = new int[t.rank];
            var lastIndexUpdated = t.rank - 3;
            var prevIndexUpdated = lastIndexUpdated;
            do
            {
                if (lastIndexUpdated <= t.rank - 3)
                {
                    if (t.rank > 2)
                        output.Append($"{String.Join(",", indices.Take(t.rank - 2))}:\n");
                    
                    output.Append("  ");
                }
                if (lastIndexUpdated == t.rank - 2)
                    output.Append("\n  ");

                // A float32 has an 8-bit exponent and a 23-bit mantissa
                // C# recommends 17 bits of precision for perfect conversion of doubles
                // and 9 bits for perfect conversion of singles.
                // However Google find usable range to be more important than precision.
                // Their bfloat16 format has an 8-bit exponent and a 7-bit mantissa,
                // this requires only ~3 digits of precision.
                // So 5 digits of precision should be enough for us.
                output.Append(t.item.ToString("g5", CultureInfo.InvariantCulture));

                indices[lastIndexUpdated] += 1;
                if (prevIndexUpdated != lastIndexUpdated)
                {
                    for (var i = lastIndexUpdated + 1; i <= prevIndexUpdated; i++)
                        indices[i] = 0;
                    
                    prevIndexUpdated = lastIndexUpdated;
                }
                t.AdvanceOffset();
                lastIndexUpdated = t.lastIndexUpdated;

                if (lastIndexUpdated == t.rank - 1 && lastIndexUpdated == prevIndexUpdated)
                    output.Append(", ");
                
            } while (lastIndexUpdated >= 0);

            return output.ToString();
        }
    }
}