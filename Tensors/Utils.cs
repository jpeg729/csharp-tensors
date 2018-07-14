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
        public static void PrintContents(this Tensor t, bool compact = false)
        {
            Console.Error.WriteLine(t);
            if (compact) {
                Console.WriteLine("\"" + t.ContentsToString(true) + "\"");
            } else {
                Console.Error.WriteLine(ContentsToString(t));
            }
        }

        public static string ContentsToString(this Tensor t, bool compact = false)
        {
            var comma = compact ? "," : ", ";
            var newline = compact ? ";" : "\n";
            var spacing = compact ? "" : "  ";
            t.Reset();
            var output = new StringBuilder(t.size * 10);
            while (t.MoveNext()) {
                if (t.lastIndexUpdated == t.rank - 1) {
                    if (t.rank > 1 || t.indices[0] > 0)
                        output.Append(comma);
                } else if (t.lastIndexUpdated == t.rank - 2) {
                    if (t.rank > 2 || t.indices[0] > 0)
                        output.Append(newline);
                    output.Append(spacing);
                } else if (t.lastIndexUpdated <= t.rank - 3) {
                    if (t.rank > 2) {
                        if (output.Length > 0)
                            output.Append(newline);
                        output.Append(String.Join(comma, t.indices.Take(t.rank - 2)));
                        output.Append(":");
                        output.Append(newline);
                    }
                    output.Append(spacing);
                }

                // A float32 has an 8-bit exponent and a 23-bit mantissa
                // C# recommends 17 bits of precision for perfect conversion of doubles
                // and 9 bits for perfect conversion of singles.
                // However Google find usable range to be more important than precision.
                // Their bfloat16 format has an 8-bit exponent and a 7-bit mantissa,
                // this requires only ~3 digits of precision.
                // So 5 digits of precision should be enough for us.
                output.Append(t.Current.ToString("g5", CultureInfo.InvariantCulture));
            }

            return output.ToString();
        }
    }
}