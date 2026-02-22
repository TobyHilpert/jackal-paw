from jaxpw.lattice.fft_grid import choose_fft_shape


def test_choose_fft_shape_even():
    assert choose_fft_shape((7, 8, 9)) == (8, 8, 10)
