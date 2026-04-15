"""
Tests unitaires pour les lois de friction.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.friction_laws import (
    power_law,
    power_law_m3,
    inverse_power_law,
    cavitation_law,
    compute_kc
)


class TestPowerLaw:
    """Tests pour la loi de puissance."""
    
    def test_power_law_basic(self):
        """Test basique de la loi de puissance."""
        u_bed = np.array([10.0, 100.0])
        m = 3.0
        As = 1e5
        
        tau = power_law(u_bed, m, As)
        
        # Vérifier que tau > 0
        assert np.all(tau > 0)
        
        # Vérifier l'ordre de grandeur
        assert tau[1] > tau[0]  # Plus de vitesse = plus de contrainte
    
    def test_power_law_m3(self):
        """Test de la loi avec m=3."""
        u_bed = np.array([10.0, 100.0])
        As = 1e5
        
        tau1 = power_law_m3(u_bed, As)
        tau2 = power_law(u_bed, 3.0, As)
        
        # Les deux méthodes doivent donner le même résultat
        assert_allclose(tau1, tau2, rtol=1e-10)
    
    def test_inverse_power_law(self):
        """Test de l'inverse de la loi de puissance."""
        tau = np.array([0.1, 1.0])
        m = 3.0
        As = 1e5
        
        # Calculer u_bed puis revenir à tau
        u_bed = inverse_power_law(tau, m, As)
        tau_back = power_law(u_bed, m, As)
        
        # Vérifier que l'inverse fonctionne
        assert_allclose(tau, tau_back, rtol=1e-10)
    
    def test_power_law_monotonic(self):
        """Test que la loi est monotone croissante."""
        u_bed = np.logspace(0, 2, 50)  # 1 à 100
        m = 3.0
        As = 1e5
        
        tau = power_law(u_bed, m, As)
        
        # Vérifier que tau est strictement croissant
        assert np.all(np.diff(tau) > 0)


class TestCavitationLaw:
    """Tests pour la loi de cavitation."""
    
    def test_cavitation_law_basic(self):
        """Test basique de la loi de cavitation."""
        u_bed = np.array([10.0, 100.0])
        CN = 1e6  # 1 MPa
        q = 2.0
        As = 1e5
        m = 3.0
        
        tau = cavitation_law(u_bed, CN, q, As, m)
        
        # Vérifier que tau > 0
        assert np.all(tau > 0)
        
        # Vérifier que tau < CN (contrainte limitée)
        assert np.all(tau <= CN)
    
    def test_cavitation_reduces_to_power_law(self):
        """Test que pour q→∞, on retrouve la loi de puissance."""
        u_bed = np.array([10.0, 50.0, 100.0])
        CN = 1e6
        q_large = 100.0  # Grand q
        As = 1e5
        m = 3.0
        
        tau_cav = cavitation_law(u_bed, CN, q_large, As, m)
        tau_pow = power_law(u_bed, m, As)
        
        # Pour grand q, on doit retrouver approximativement la loi de puissance
        # (tant que tau << CN)
        # Note: ce test est approximatif
        relative_diff = np.abs((tau_cav - tau_pow) / tau_pow)
        assert np.all(relative_diff < 0.1)  # Moins de 10% de différence


class TestComputeKc:
    """Tests pour le calcul de k_c."""
    
    def test_compute_kc_basic(self):
        """Test basique du calcul de k_c."""
        Q = 0.1
        kc = compute_kc(Q)
        
        # Vérifier que k_c > 0
        assert kc > 0
        
        # Vérifier l'ordre de grandeur (doit être très grand)
        assert kc > 1e10
    
    def test_compute_kc_monotonic(self):
        """Test que k_c décroît quand Q augmente."""
        Q_vals = np.array([0.01, 0.1, 1.0, 10.0])
        kc_vals = np.array([compute_kc(Q) for Q in Q_vals])
        
        # k_c doit décroître avec Q (car Q^(-(alpha_c - 1)))
        # alpha_c = 1.25, donc exposant négatif
        assert np.all(np.diff(kc_vals) < 0)


class TestConstants:
    """Tests pour les constantes."""
    
    def test_constants_import(self):
        """Test que les constantes peuvent être importées."""
        from src.constants import RHO_I, G, L, N, A
        
        assert RHO_I == 910.0
        assert G == 9.81
        assert L == 3.35e5
        assert N == 3.0
        assert A == 2.4e-24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
