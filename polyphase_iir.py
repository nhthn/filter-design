import numpy


def _next_odd_integer(x):
    x = int(numpy.ceil(x))
    x = x + (x + 1) % 2
    return x


def design_optimal_halfband(
        transition_bandwidth=0.2,
        target_attenuation_db=100,
        allpass_count=None
        ):
    """
    Computes the allpass coefficients of an optimal polyphase IIR filter with a
    cutoff of 1/2 Nyquist.

    The transition bandwidth is specified as a fraction of 2pi.
    """
    transition_bandwidth = transition_bandwidth * 2 * numpy.pi

    k = numpy.tan((numpy.pi - transition_bandwidth) / 4) ** 2
    k_prime = numpy.sqrt(1 - k**2)
    e = (1 - numpy.sqrt(k_prime)) / (1 + numpy.sqrt(k_prime)) * 0.5
    q = e * (1 + 2 * e**4 + 15 * e**8 + 150 * e**12)

    if allpass_count is None:
        target_attenuation = 10 ** (-target_attenuation_db / 20)
        k1 = (target_attenuation**2) / (1 - target_attenuation**2)
        n = _next_odd_integer(numpy.log(k1**2 / 16) / numpy.log(q))
        allpass_count = n // 2
    else:
        n = allpass_count * 2 + 1

    q1 = q ** n
    k1 = 4 * numpy.sqrt(q1)
    attenuation = numpy.sqrt(k1 / (1 + k1))
    passband_ripple = 1 - numpy.sqrt(1 - attenuation**2)

    i = numpy.arange(1, allpass_count + 1)
    tmp_numerator = 0
    tmp_denominator = 0
    for m in range(5):
        sign = -1 if m % 2 == 1 else 1
        tmp_numerator += (
            sign * q**(m * (m + 1))
            * numpy.sin((2 * m + 1) * numpy.pi * i / n)
            )
        if m > 0:
            tmp_denominator += (
                sign * q**(m * m)
                * numpy.cos(2 * m * numpy.pi * i / n)
                )
    w = 2 * q ** 0.25 * tmp_numerator / (1 + 2 * tmp_denominator)
    a_prime = numpy.sqrt((1 - w * w * k) * (1 - w * w / k)) / (1 + w * w)
    coefficients = (1 - a_prime) / (1 + a_prime)
    path_1_coefficients = coefficients[0::2]
    path_2_coefficients = coefficients[1::2]

    return {
        "coefficients": coefficients,
        "path_1_coefficients": path_1_coefficients,
        "path_2_coefficients": path_2_coefficients,
        "filter_order": n,
        "allpass_count": allpass_count,
        "attenuation_db": 20 * numpy.log10(attenuation),
        "passband_ripple_db": 20 * numpy.log10(1 + passband_ripple),
        }
