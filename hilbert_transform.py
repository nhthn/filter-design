"""
Weaver, Donald K. 1953. "Design of RC Wide-Band 90-Degree Phase-Difference
Network."
Hutchins, Bernie. "The Design of Wideband Analog 90-Degree Phase Differencing
Networks Without a Large Spread of Capacitor Values."
"""
import numpy
import numpy.polynomial.polynomial as poly


class PhaseDifferenceNetwork:
    def __init__(self, poles_a, poles_b, error_in_degrees):
        self.poles_a = poles_a
        self.poles_b = poles_b
        self.allpass_count = len(self.poles_a) + len(self.poles_b)
        self.error_in_degrees = error_in_degrees

    def transfer_function(self, frequencies):
        """Compute the transfer function at a given angular frequency.
        You can also pass in a NumPy array.
        """
        s = frequencies * 1j

        def allpass(s, a):
            return (s - a) / (s + a)

        H_a = 1
        for pole in self.poles_a:
            H_a *= allpass(s, pole)
        H_b = 1
        for pole in self.poles_b:
            H_b *= allpass(s, pole)

        return (H_a, H_b)

    def phase(self, frequencies):
        """Compute the unwrapped phase at a given frequency. Avoids numerical
        unwrapping by using an analytical formula for the phase contribution of
        a one-pole one-zero allpass filter along the imaginary axis:

        H = (s + p) / (s - p)
        phase(w) = pi + 2 arctan(w / p)

        See https://ccrma.stanford.edu/realsimple/DelayVar/Phasing_First_Order_Allpass_Filters.html.
        Note that the reference uses the "break frequency" w_b = -p, which means
        we have a sign flip relative to the reference.
        """
        def phase(w, a):
            return numpy.rad2deg(numpy.pi + 2 * numpy.arctan(w / a))

        # Add multiples of 2pi so that the graph is close to 0 at DC
        offset = -numpy.sum(phase(0, self.poles_a))
        offset = 2 * numpy.pi * round(offset / (2 * numpy.pi))

        phase_a = offset
        for pole in self.poles_a:
            phase_a += phase(frequencies, pole)
        phase_b = offset
        for pole in self.poles_b:
            phase_b += phase(frequencies, pole)
        phase_difference = phase_a - phase_b

        return phase_a, phase_b, phase_difference

    def plot(self):
        """Plot the phase responses with matplotlib.
        """
        import matplotlib.pyplot as plt

        plt.semilogx()
        plt.xlim(1.0, 100e3)
        plt.title(f"{self.allpass_count}-pole 90-degree phase-difference network")
        plt.xlabel("Angular frequency (radians/sec)")
        plt.ylabel("Phase (deg.)")
        #plt.ylim(-180, 180)
        w = numpy.geomspace(1, 100e3, 2000, endpoint=False)
        phase_a, phase_b, phase_difference = self.phase(w)
        plt.plot(w, phase_a, label="Network A phase response")
        plt.plot(w, phase_b, label="Network B phase response")
        plt.plot(w, phase_difference, label="Phase difference")
        plt.show()

    def print(self):
        """Print all filter coefficients to full precision, as well as some
        human-readable technical specs.
        """
        print("90-degree phase differencing network")
        print("Poles, path A: " + ", ".join([x.astype(str) for x in self.poles_a]))
        print("Poles, path B: " + ", ".join([x.astype(str) for x in self.poles_b]))
        print(f"Error: {self.error_in_degrees:.4}°")

def design_weaver_method(min_frequency, max_frequency, n):
    k_prime = min_frequency / max_frequency
    k = numpy.sqrt(1 - k_prime * k_prime)
    sqrt_k = numpy.sqrt(k)
    ell = 0.5 * (1 - sqrt_k) / (1 + sqrt_k)
    q_prime = ell + 2 * (ell ** 5) + 15 * (ell ** 9)
    q = numpy.exp(numpy.pi * numpy.pi / numpy.log(q_prime))

    if n % 2 == 0:
        r_a = r_b = numpy.arange(1, n // 2 + 1)
    else:
        r_a = numpy.arange(1, (n + 1) // 2 + 1)
        r_b = numpy.arange(1, (n - 1) // 2 + 1)
    phi_a = numpy.pi / (4 * n) * (4 * r_a - 3)
    phi_b = numpy.pi / (4 * n) * (4 * r_b - 1)
    phi_a_prime = numpy.arctan(
        (q ** 2 - q ** 6) * numpy.sin(4 * phi_a)
        / (1 + (q ** 2 + q ** 6) * numpy.cos(4 * phi_a))
    )
    phi_b_prime = numpy.arctan(
        (q ** 2 - q ** 6) * numpy.sin(4 * phi_b)
        / (1 + (q ** 2 + q ** 6) * numpy.cos(4 * phi_b))
    )
    poles_a = numpy.tan(phi_a - phi_a_prime) / numpy.sqrt(k_prime)
    poles_b = numpy.tan(phi_b - phi_b_prime) / numpy.sqrt(k_prime)

    return poles_a, poles_b


def design_weaver_method_2(
    min_frequency,
    max_frequency,
    *,
    n=None,
    max_error_in_degrees=None
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
        raise TypeError("Both 'n' and 'max_error_in_degrees' are specified. You can only specify one.")
    elif n is not None and max_error_in_degrees is None:
        # All OK.
        pass
    elif n is None and max_error_in_degrees is not None:
        n = int(numpy.ceil(numpy.log(max_error_in_degrees * numpy.pi / 720) / ln_q))
    error_in_degrees = 720 * q ** n / numpy.pi

    r = numpy.arange(1, n + 1)
    phi = numpy.pi / (4 * n) * (2 * r - 1)
    phi_prime = numpy.arctan(
        (q ** 2 - q ** 6) * numpy.sin(4 * phi)
        / (1 + (q ** 2 + q ** 6) * numpy.cos(4 * phi))
    )
    poles = -min_frequency * numpy.tan(phi - phi_prime) / numpy.sqrt(k_prime)
    poles_a = poles[0::2]
    poles_b = poles[1::2]

    return PhaseDifferenceNetwork(poles_a, poles_b, error_in_degrees)

if __name__ == "__main__":
    filter_ = design_weaver_method_2(20, 20000, n=12)
    filter_.print()
    filter_.plot()
