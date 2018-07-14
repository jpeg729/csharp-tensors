using System;
using System.Text;
using System.Linq;
using System.Globalization;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Tensors
{
    
    public static class Utils
    {
        public static void PrintContents(this Tensor t)
        {
            Console.Error.WriteLine(t);
            Console.Error.WriteLine(ContentsToString(t));
        }

        public static string ContentsToString(this Tensor t)
        {
            t.ResetOffset();
            var output = new StringBuilder(t.size * 10);
            while (true)
            {
                if (t.lastIndexUpdated <= t.rank - 3)
                {
                    if (t.rank > 2)
                    {
                        if (output.Length > 0)
                            output.Append("\n");
                        output.Append($"{String.Join(",", t.indices.Take(t.rank - 2))}:\n");
                    }
                    output.Append("  ");
                }
                if (t.lastIndexUpdated == t.rank - 2)
                    output.Append("\n  ");

                // A float32 has an 8-bit exponent and a 23-bit mantissa
                // C# recommends 17 bits of precision for perfect conversion of doubles
                // and 9 bits for perfect conversion of singles.
                // However Google find usable range to be more important than precision.
                // Their bfloat16 format has an 8-bit exponent and a 7-bit mantissa,
                // this requires only ~3 digits of precision.
                // So 5 digits of precision should be enough for us.
                output.Append(t.item.ToString("g5", CultureInfo.InvariantCulture));

                try
                {
                    if (!t.AdvanceOffset())
                        break;
                }
                catch
                {
                    Console.WriteLine(output);
                    Console.WriteLine(Environment.StackTrace);
                    return "Error while reading contents";
                }

                if (t.lastIndexUpdated == t.rank - 1)
                    output.Append(", ");
            }

            return output.ToString();
        }
    }
}