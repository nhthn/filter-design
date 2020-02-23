"""
Derives the coefficients for an optimal half-band polyphase IIR filter with two paths
of allpass filters and provides some basic metrics on the filter's response. The
resulting set of coefficients is independent of sample rate.

Algorithm from Valenzuela, R. A. and Constantinides, A. G. 1983, "Digital signal
processing schemes for efficient interpolation and decimation."
"""
import numpy


class HalfbandPolyphaseIIRFilter:
    def __init__(self, coefficients, design_specific_info=None):
        self.coefficients = coefficients
        self.paths = len(coefficients)
        self.allpass_count = sum([len(path) for path in self.coefficients])
        self.design_specific_info = design_specific_info

    def transfer_function(self, frequencies):
        """Compute the transfer function at a given frequency (angular, from 0
        to 2pi). You can also pass in a NumPy array.
        """
        z = numpy.exp(frequencies * 1j)

        def allpass(z, a):
            return (a + z ** -2) / (1 + a * z ** -2)

        H = 0
        for i, path in enumerate(self.coefficients):
            H_path = z ** -i
            for a in path:
                H_path *= allpass(z, a)
            H += H_path
        H /= self.paths

        return H

    def gain(self, frequencies):
        """Compute the gain in dB at a given frequency (angular, from 0 to
        2pi). You can also pass in a NumPy array.
        """
        return 20 * numpy.log10(numpy.abs(self.transfer_function(frequencies)))

    def plot(self):
        """Plot the magnitude response with matplotlib.
        """
        import matplotlib.pyplot as plt

        w = numpy.linspace(0, numpy.pi, 1000, endpoint=False)
        plt.title("{}-allpass polyphase IIR filter".format(self.allpass_count))
        plt.xlabel("Angular frequency (radians)")
        plt.ylabel("Gain (dB)")
        plt.ylim(-150, 5)
        plt.plot(w, self.gain(w))
        plt.show()

    def passband_edge(self, tolerance=-1.0):
        """Perform a binary search to identify the frequency where the gain of
        the filter crosses a certain value. This is used to get accurate
        technical specs on the passband such as "-1 dB from 0 to 21 kHz".
        """
        left = 0
        right = numpy.pi

        for i in range(100):
            center = (left + right) * 0.5
            gain = self.gain(center)
            if gain < tolerance:
                right = center
            else:
                left = center

        return center

    def print(self):
        """Print all filter coefficients to full precision, as well as some
        human-readable technical specs.
        """
        print("Coefficients:")
        for path in self.coefficients:
            print(", ".join([x.astype(str) for x in path]))
        print()
        print("Paths: {}".format(self.paths))
        print("Total allpasses: {}".format(self.allpass_count))

        if self.design_specific_info["type"] == "optimal halfband":
            print()
            print("Optimal halfband design")
            print(
                "Stopband attenuation: {:.4} dB".format(
                    self.design_specific_info["attenuation_db"]
                )
            )
            print(
                "Passband ripple: {:.4} dB".format(
                    self.design_specific_info["passband_ripple_db"]
                )
            )

            tolerance = -0.01
            edge = self.passband_edge(tolerance) / numpy.pi
            print(
                "Passband: {:.4} dB from 0 to {:.4} * Nyquist ({} Hz at Fs = 44100)".format(
                    tolerance, edge, int(edge * 44100)
                )
            )


def _next_odd_integer(x):
    x = int(numpy.ceil(x))
    x = x + (x + 1) % 2
    return x


def design_optimal_halfband(
    transition_bandwidth=0.2, target_attenuation_db=-100, allpass_count=None
):
    """
    Computes the allpass coefficients of an optimal polyphase IIR filter with a
    cutoff of 1/2 Nyquist.

    The transition bandwidth is specified as a fraction of 2pi.
    """
    transition_bandwidth = transition_bandwidth * 2 * numpy.pi

    k = numpy.tan((numpy.pi - transition_bandwidth) / 4) ** 2
    k_prime = numpy.sqrt(1 - k ** 2)
    e = (1 - numpy.sqrt(k_prime)) / (1 + numpy.sqrt(k_prime)) * 0.5
    q = e * (1 + 2 * e ** 4 + 15 * e ** 8 + 150 * e ** 12)

    if allpass_count is None:
        target_attenuation = 10 ** (target_attenuation_db / 20)
        k1 = (target_attenuation ** 2) / (1 - target_attenuation ** 2)
        n = _next_odd_integer(numpy.log(k1 ** 2 / 16) / numpy.log(q))
        allpass_count = n // 2
    else:
        n = allpass_count * 2 + 1

    q1 = q ** n
    k1 = 4 * numpy.sqrt(q1)
    attenuation = numpy.sqrt(k1 / (1 + k1))
    passband_ripple = 1 - numpy.sqrt(1 - attenuation ** 2)

    i = numpy.arange(1, allpass_count + 1)
    tmp_numerator = 0
    tmp_denominator = 0
    for m in range(5):
        sign = -1 if m % 2 == 1 else 1
        tmp_numerator += (
            sign * q ** (m * (m + 1)) * numpy.sin((2 * m + 1) * numpy.pi * i / n)
        )
        if m > 0:
            tmp_denominator += sign * q ** (m * m) * numpy.cos(2 * m * numpy.pi * i / n)
    w = 2 * q ** 0.25 * tmp_numerator / (1 + 2 * tmp_denominator)
    a_prime = numpy.sqrt((1 - w * w * k) * (1 - w * w / k)) / (1 + w * w)
    coefficients = (1 - a_prime) / (1 + a_prime)

    return HalfbandPolyphaseIIRFilter(
        [coefficients[0::2], coefficients[1::2]],
        {
            "type": "optimal halfband",
            "attenuation_db": 20 * numpy.log10(attenuation),
            "passband_ripple_db": 20 * numpy.log10(1 + passband_ripple),
        },
    )


if __name__ == "__main__":
    the_filter = design_optimal_halfband(0.15, -90)
    the_filter.print()
    the_filter.plot()
