import unittest

from pyedautils import units
from pyedautils._mollier import state


class TestTemperature(unittest.TestCase):

    def test_freezing_point(self):
        self.assertAlmostEqual(units.c_to_f(0.0), 32.0, places=6)
        self.assertAlmostEqual(units.f_to_c(32.0), 0.0, places=6)

    def test_boiling_point(self):
        self.assertAlmostEqual(units.c_to_f(100.0), 212.0, places=6)
        self.assertAlmostEqual(units.f_to_c(212.0), 100.0, places=6)

    def test_round_trip(self):
        for t in [-20.0, 0.0, 20.0, 50.0, 100.0]:
            self.assertAlmostEqual(units.f_to_c(units.c_to_f(t)), t, places=10)

    def test_kelvin(self):
        self.assertAlmostEqual(units.c_to_k(0.0), 273.15, places=8)
        self.assertAlmostEqual(units.k_to_c(273.15), 0.0, places=8)


class TestHumidityRatio(unittest.TestCase):

    def test_grlb_known(self):
        # 1 lb dry air × 7000 grains/lb = 7000 gr/lb if x=1 (impossible, but
        # the multiplier is exact). For x=0.01: 70 gr/lb.
        self.assertAlmostEqual(units.kgkg_to_grlb(0.01), 70.0, places=6)

    def test_round_trip(self):
        for x in [0.001, 0.005, 0.010, 0.025]:
            self.assertAlmostEqual(
                units.grlb_to_kgkg(units.kgkg_to_grlb(x)), x, places=10)

    def test_gkg_known(self):
        self.assertAlmostEqual(units.kgkg_to_gkg(0.012), 12.0, places=8)
        self.assertAlmostEqual(units.gkg_to_kgkg(15.0), 0.015, places=8)


class TestMassFlow(unittest.TestCase):

    def test_kg_per_lb(self):
        self.assertAlmostEqual(units.kgs_to_lbh(0.5), 3968.32, delta=0.01)

    def test_round_trip(self):
        for m in [0.1, 0.5, 1.0, 10.0]:
            self.assertAlmostEqual(
                units.lbh_to_kgs(units.kgs_to_lbh(m)), m, places=10)

    def test_kgs_to_kgh(self):
        self.assertAlmostEqual(units.kgs_to_kgh(1.0), 3600.0, places=8)
        self.assertAlmostEqual(units.kgh_to_kgs(3600.0), 1.0, places=8)


class TestPower(unittest.TestCase):

    def test_known(self):
        # 1 kW = 3412.14 BTU/h (NIST)
        self.assertAlmostEqual(units.kw_to_btuh(1.0), 3412.14, delta=0.05)

    def test_round_trip(self):
        for p in [0.5, 1.0, 5.0, 100.0]:
            self.assertAlmostEqual(
                units.btuh_to_kw(units.kw_to_btuh(p)), p, places=10)


class TestPressure(unittest.TestCase):

    def test_atmospheric_psi(self):
        # 1 atm = 101325 Pa = 14.696 psi
        self.assertAlmostEqual(units.pa_to_psi(101325.0), 14.696, delta=0.001)

    def test_atmospheric_inhg(self):
        # 1 atm ≈ 29.92 inHg
        self.assertAlmostEqual(units.pa_to_inhg(101325.0), 29.92, delta=0.01)

    def test_kpa_round_trip(self):
        for p in [50000, 80000, 101325, 110000]:
            self.assertAlmostEqual(units.kpa_to_pa(units.pa_to_kpa(p)), p,
                                   places=8)

    def test_psi_round_trip(self):
        for p in [50000, 80000, 101325]:
            self.assertAlmostEqual(units.psi_to_pa(units.pa_to_psi(p)), p,
                                   places=6)


class TestVolumeFlow(unittest.TestCase):

    def test_known(self):
        # 1000 m³/h ≈ 588.58 CFM
        self.assertAlmostEqual(units.m3h_to_cfm(1000.0), 588.58, delta=0.05)

    def test_round_trip(self):
        for v in [100.0, 500.0, 1500.0, 10000.0]:
            self.assertAlmostEqual(
                units.cfm_to_m3h(units.m3h_to_cfm(v)), v, places=8)


class TestLength(unittest.TestCase):

    def test_meter_foot(self):
        self.assertAlmostEqual(units.m_to_ft(1.0), 3.28084, delta=1e-5)
        self.assertAlmostEqual(units.ft_to_m(3.28084), 1.0, delta=1e-5)


class TestEnthalpy(unittest.TestCase):

    def test_known(self):
        # 1 BTU/lb = 2.326 kJ/kg
        self.assertAlmostEqual(units.btulb_to_kjkg(1.0), 2.326, places=6)
        self.assertAlmostEqual(units.kjkg_to_btulb(2.326), 1.0, places=6)


class TestDensity(unittest.TestCase):

    def test_known(self):
        # Standard dry air at 20 °C ≈ 1.204 kg/m³ ≈ 0.0751 lb/ft³
        self.assertAlmostEqual(units.kgm3_to_lbft3(1.204), 0.0752, delta=0.001)


class TestSpecificVolume(unittest.TestCase):

    def test_round_trip(self):
        for v in [0.8, 0.83, 0.86]:
            self.assertAlmostEqual(
                units.ft3lb_to_m3kg(units.m3kg_to_ft3lb(v)), v, places=10)


class TestStateToIp(unittest.TestCase):

    def test_returns_dict_with_expected_keys(self):
        s = state(t=25.0, phi=0.5, p=101325.0)
        ip = units.state_to_ip(s)
        expected = {
            't_F', 't_wb_F', 't_dp_F', 'phi_pct',
            'x_grlb', 'h_btulb', 'p_v_inhg', 'rho_lbft3',
            'v_ft3lb', 'p_psi', 'm_dot_dry_lbh',
            'volume_flow_cfm', 'convention',
        }
        self.assertEqual(set(ip.keys()), expected)

    def test_temperature_converted(self):
        s = state(t=20.0, phi=0.5, p=101325.0)
        ip = units.state_to_ip(s)
        self.assertAlmostEqual(ip['t_F'], 68.0, delta=0.01)

    def test_phi_as_percent(self):
        s = state(t=20.0, phi=0.42, p=101325.0)
        ip = units.state_to_ip(s)
        self.assertAlmostEqual(ip['phi_pct'], 42.0, places=5)

    def test_pressure_psi(self):
        s = state(t=20.0, phi=0.5, p=101325.0)
        ip = units.state_to_ip(s)
        self.assertAlmostEqual(ip['p_psi'], 14.696, delta=0.001)

    def test_mass_flow_present_when_set(self):
        s = state(t=20.0, phi=0.5, p=101325.0, m_dot_dry=0.5)
        ip = units.state_to_ip(s)
        self.assertIsNotNone(ip['m_dot_dry_lbh'])

    def test_mass_flow_none_when_unset(self):
        s = state(t=20.0, phi=0.5, p=101325.0)
        ip = units.state_to_ip(s)
        self.assertIsNone(ip['m_dot_dry_lbh'])
        self.assertIsNone(ip['volume_flow_cfm'])

    def test_volume_flow_converted(self):
        # 1000 m³/h ≈ 588.58 CFM
        s = state(t=20.0, phi=0.5, p=101325.0, volume_flow=1000.0)
        ip = units.state_to_ip(s)
        self.assertAlmostEqual(ip['volume_flow_cfm'], 588.58, delta=0.05)

    def test_convention_passes_through(self):
        s = state(t=20.0, phi=0.5, p=101325.0, convention='glueck')
        ip = units.state_to_ip(s)
        self.assertEqual(ip['convention'], 'glueck')


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
