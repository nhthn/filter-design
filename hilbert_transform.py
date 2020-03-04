"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
"""
import numpy
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


class PhaseDifferenceNetwork:
    def __init__(
        self, poles_a, poles_b, min_frequency, max_frequency, error_in_degrees
    ):
        self.poles_a = poles_a
        self.poles_b = poles_b
        self.allpass_count = len(self.poles_a) + len(self.poles_b)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.error_in_degrees = error_in_degrees

    @classmethod
    def design(
        cls, min_frequency, max_frequency, *, n=None, max_error_in_degrees=None
    ):
        k_prime = min_frequency / max_frequency
        k = numpy.sqrt(1 - k_prime * k_prime)
        sqrt_k = numpy.sqrt(k)
        ell = 0.5 * (1 - sqrt_k) / (1 + sqrt_k)
        q_prime = ell + 2 * (ell ** 5) + 15 * (ell ** 9)
        ln_q = numpy.pi ** 2 / numpy.log(q_prime)
        q = numpy.exp(ln_q)

        if n is None and max_error_in_degrees is None:
            raise TypeError("Please specify either 'n' or 'max_error_in_degrees'.")
        elif n is not None and max_error_in_degrees is not None:
            raise TypeError(
                "Both 'n' and 'max_error_in_degrees' are specified. You can only specify one."
            )
        elif n is not None and max_error_in_degrees is None:
            # All OK.
            pass
        elif n is None and max_error_in_degrees is not None:
            n = int(numpy.ceil(numpy.log(max_error_in_degrees * numpy.pi / 720) / ln_q))
        error_in_degrees = 720 * q ** n / numpy.pi

        r = numpy.arange(1, n + 1)
        phi = numpy.pi / (4 * n) * (2 * r - 1)
        phi_prime = numpy.arctan(
            (q ** 2 - q ** 6)
            * numpy.sin(4 * phi)
            / (1 + (q ** 2 + q ** 6) * numpy.cos(4 * phi))
        )
        poles = (
            -2 * numpy.pi * min_frequency * numpy.tan(phi - phi_prime) / numpy.sqrt(k_prime)
        )
        poles_a = poles[0::2]
        poles_b = poles[1::2]

        return cls(
            poles_a, poles_b, min_frequency, max_frequency, error_in_degrees
        )

    def phase(self, frequencies):
        """Compute the unwrapped phase at a given frequency. Avoids numerical
        unwrapping by using an analytical formula for the phase contribution of
        a one-pole one-zero allpass filter along the imaginary axis:

        H = (s + p) / (s - p)
        phase(w) = pi + 2 arctan(w / p)

        See https://ccrma.stanford.edu/realsimple/DelayVar/Phasing_First_Order_Allpass_Filters.html.
        Note that the reference uses the "break frequency" w_b = -p, which means
        we have a sign flip relative to them.
        """

        def phase(w, a):
            return numpy.pi + 2 * numpy.arctan(w / a)

        # Add multiples of 2pi so that the graph is close to 0 at DC
        offset = -numpy.sum(phase(0, self.poles_a))
        offset = 2 * numpy.pi * round(offset / (2 * numpy.pi))

        w = 2 * numpy.pi * frequencies

        phase_a = offset
        for pole in self.poles_a:
            phase_a += phase(w, pole)
        phase_b = offset
        for pole in self.poles_b:
            phase_b += phase(w, pole)

        return phase_a, phase_b

    def plot(self, mode="phase_responses"):
        """Plot the phase responses or differences with matplotlib."""
        figure = plt.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.set_xscale("log")
        axes.set_xlim(1.0, 100e3)
        title_part_1 = f"{self.allpass_count}-pole 90-degree phase-difference network"
        axes.set_xlabel("Angular frequency (radians/sec)")
        axes.set_ylabel("Phase (deg.)")
        # figure.ylim(-180, 180)
        w = numpy.geomspace(1, 100e3, 2000, endpoint=False)
        phase_a, phase_b = self.phase(w)
        if mode == "phase_responses":
            figure.suptitle(f"{title_part_1}, phase responses")
            axes.plot(w, numpy.rad2deg(phase_a), label="Allpass A phase response")
            axes.plot(w, numpy.rad2deg(phase_b), label="Allpass B phase response")
        elif mode == "phase_difference":
            figure.suptitle(f"{title_part_1}, phase difference")
            phase_difference = phase_a - phase_b
            axes.plot(w, numpy.rad2deg(phase_difference))
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return figure

    def get_average_group_delay(self):
        """Compute the group delay of each network over the specified bandwidth, averaged
        over the bandwidth and averaged across the two networks.

        This uses a shortcut employing the Fundamental Theorem of Calculus. Since phase
        response is the antiderivative of group delay (aside from a sign flip), this average
        can be found by simply computing the phase response at the two endpoints of the
        interval and dividing by the difference between the endpoints. In formula:

        f1 = min angular frequency
        f2 = max angular frequency
        average group delay = -(phase_response(f2) - phase_response(f1)) / (f2 - f1)

        The group delay is in seconds, or at least I think so. (If you walk through what
        happens if H = e^(sT), you get a group delay of T.)
        """
        endpoints = numpy.array([self.min_frequency, self.max_frequency])
        # The endpoints are here specified in frequency. When computing this denominator,
        # the interval size must be converted to angular frequency, hence multiplication by
        # 2pi.
        k = 1 / ((endpoints[1] - endpoints[0]) * 2 * numpy.pi)
        phase_a, phase_b = self.phase(endpoints)
        group_delay_a = -(phase_a[1] - phase_a[0]) * k
        group_delay_b = -(phase_b[1] - phase_b[0]) * k
        average_group_delay = (group_delay_a + group_delay_b) * 0.5
        return average_group_delay

    def print_info(self):
        """Print all filter coefficients to full precision, as well as some
        human-readable technical specs.
        """
        print("90-degree phase differencing network")
        print("Poles, path A:")
        for pole in self.poles_a:
            print(f"    {pole.astype(str)}")
        print("Poles, path B:")
        for pole in self.poles_b:
            print(f"    {pole.astype(str)}")
        print(f"Error: {self.error_in_degrees:.4}°")
        print()

        average_group_delay = self.get_average_group_delay()
        print(
            f"Average group delay over specified bandwidth: {average_group_delay * 1e3:.4}ms"
        )


if __name__ == "__main__":
    filter_ = PhaseDifferenceNetwork.design(20.0, 20000.0, n=12)
    filter_.print_info()
    figure = filter_.plot("phase_responses")
    figure.savefig("frequency_shifter_phase_response.png")
    figure = filter_.plot("phase_difference")
    figure.savefig("frequency_shifter_phase_difference.png")
